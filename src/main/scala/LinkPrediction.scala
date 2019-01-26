import org.apache.spark.SparkConf
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.graphx._
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, VectorAssembler}
import org.apache.spark.ml.linalg
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark._
import scala.collection.mutable
import org.apache.spark.ml.linalg.Vectors

//import org.apache.spark.ml.feature.MinHashLSH
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

object LinkPrediction {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Program Files\\JetBrains\\IntelliJ IDEA 2018.3\\")
    val master = "local[*]"
    val appName = "Link Prediction"
    // Create the spark session first
    val conf: SparkConf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)
      .set("spark.driver.allowMultipleContexts", "false")
      .set("spark.scheduler.allocation.file", "src/resources/fairscheduler.xml")
      .set("spark.scheduler.pool", "default")
      .set("spark.scheduler.mode", "FAIR")
      .set("spark.ui.enabled", "true")
      .set("spark.memory.fraction", "1")
      .set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
      .set("spark.default.parallelism", "100")
//      .set("spark.speculation", "false")
//      .set("spark.speculation.quantile", "0.75")
//      .set("spark.speculation.multiplier", "1.5")
      .set("spark.sql.shuffle.partitions", "40") //Number of partitions = Total input dataset size / partition size => 1500 / 64 = 23.43 = ~23 partitions.
//      .set("spark.task.cpus", "1")
//      .set("spark.dynamicAllocation.enabled", "true")
//      .set("spark.dynamicAllocation.minExecutors", "1")
//      .set("spark.dynamicAllocation.executorAllocationRatio", "1")
//      .set("spark.streaming.backpressure.enabled", "true")
 //     .set("spark.streaming.blockInterval", "1000ms")
    val ss: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames
    // Suppress info messages
    ss.sparkContext.setLogLevel("ERROR")
    val currentDir = System.getProperty("user.dir") // get the current directory
    val trainingSetFile = "./training_set.txt"
    val testingSetFile = "./testing_set.txt"
    val nodeInfoFile = "./node_information.csv"
    val groundTruthNetworkFile = "./Cit-HepTh.txt"

    def toDoubleUDF = udf(
      (n: Int) => n.toDouble
    )

    def cosineSimilarityUDF = udf(
      (a: linalg.Vector, b: linalg.Vector) => {
        val emptyVec = Vectors.zeros(a.size)
        if (a.equals(emptyVec) || b.equals(emptyVec)) {
          0.0
        } else {
          val aArr = a.toArray
          val bArr = b.toArray
          val dotProduct = aArr.zip(bArr).map(t => t._1 * t._2).sum
          dotProduct / Math.sqrt(Vectors.sqdist(a, emptyVec) * Vectors.sqdist(b, emptyVec))
        }
      }
    )

    def myUDF: UserDefinedFunction = udf(
      (s1: String, s2: String) => {
        val splitted1 = s1.split(",")
        val splitted2 = s2.split(",")
        splitted1.intersect(splitted2).length
      })

    def commonNeighbors = udf(
      (a: mutable.WrappedArray[Long], b: mutable.WrappedArray[Long]) =>
        a.intersect(b).length
    )

    def jaccardCoefficient = udf(
      (a: mutable.WrappedArray[Long], b: mutable.WrappedArray[Long]) =>
        a.intersect(b).length.toDouble / a.union(b).distinct.length.toDouble
    )

    def transformGraph(graph: Graph[Int, Int]): DataFrame = {
      val inDegreesDF = graph.inDegrees.toDF("id", "inDegrees")

      val commonNeighborsDF = graph.ops.collectNeighborIds(EdgeDirection.Either)
        .toDF("id", "neighbors")
        .cache()

      inDegreesDF.join(commonNeighborsDF, "id")
    }

    def transformNodeInfo(input: DataFrame): DataFrame = {
      // Create tf-idf features
      val tokenizer = new Tokenizer().setInputCol("abstract").setOutputCol("abstractWords") //STOP - WORDS REMOVAL
      val wordsDF = tokenizer.transform(input.na.fill(Map("abstract" -> "")))

      val hashingTF = new HashingTF().setInputCol("abstractWords").setOutputCol("abstractRawFeatures").setNumFeatures(20000)
      val featurizedDF = hashingTF.transform(wordsDF)

      val idf = new IDF().setInputCol("abstractRawFeatures").setOutputCol("abstractFeatures")
      val idfM = idf.fit(featurizedDF)
      val tfidfDF = idfM.transform(featurizedDF)

      val numOfClusters = 6
      val kMeans = new KMeans().setK(numOfClusters).setFeaturesCol("abstractFeatures").setSeed(1L)
      val model = kMeans.fit(tfidfDF).setPredictionCol("clusterCenter")
      val completeDF = model.transform(tfidfDF)

      completeDF
    }

    def transformSet(input: DataFrame, nodeInfo: DataFrame, graph: DataFrame): DataFrame = {
      val assembler = new VectorAssembler()
        //        .setInputCols(Array("yearDiff", "isSameJournal","cosSimTFIDF"))
        .setInputCols(Array("yearDiff", "nCommonAuthors", "isSelfCitation", "isSameJournal", "cosSimTFIDF", "tInDegrees", "inDegreesDiff", "commonNeighbors", "jaccardCoefficient", "InSameCluster"))
        .setOutputCol("features")

      val tempDF = input
        .join(nodeInfo
          .select($"id",
            $"authors".as("sAuthors"),
            $"year".as("sYear"),
            $"journal".as("sJournal"),
            $"clusterCenter".as("sCluster"),
            $"abstractFeatures".as("sAbstractFeatures")), $"sId" === $"id")
        .drop("id")

      val tempDF2 = tempDF
        .join(nodeInfo
          .select($"id",
            $"authors".as("tAuthors"),
            $"year".as("tYear"),
            $"journal".as("tJournal"),
            $"clusterCenter".as("tCluster"),
            $"abstractFeatures".as("tAbstractFeatures")), $"tId" === $"id")
        .drop("id")
        .withColumn("yearDiff", $"tYear" - $"sYear")
        .withColumn("nCommonAuthors", when($"sAuthors".isNotNull && $"tAuthors".isNotNull, myUDF('sAuthors,'tAuthors)).otherwise(0))
        .withColumn("isSelfCitation", $"nCommonAuthors" >= 1)
        .withColumn("isSameJournal", when($"sJournal" === $"tJournal", true).otherwise(false))
        .withColumn("InSameCluster", when($"sCluster" === $"tCluster", true).otherwise(false))
        .withColumn("cosSimTFIDF", bround(cosineSimilarityUDF($"sAbstractFeatures", $"tAbstractFeatures"),3))
        .drop("sAbstractFeatures").drop("tAbstractFeatures")

      val tempDF3 = tempDF2
        .join(graph
          .select($"id",
            $"inDegrees".as("sInDegrees"),
            $"neighbors".as("sNeighbors")), $"sId" === $"id", "left")
        .na.fill(Map("sInDegrees" -> 1))
        .drop("id")

      val tempDF4 = tempDF3
        .join(graph
          .select($"id",
            $"inDegrees".as("tInDegrees"),
            $"neighbors".as("tNeighbors")), $"tId" === $"id", "left")
        .na.fill(Map("tInDegrees" -> 1))
        .drop("id")
        .withColumn("inDegreesDiff", $"tInDegrees" - $"sInDegrees")
        .withColumn("commonNeighbors", when($"sNeighbors".isNotNull && $"tNeighbors".isNotNull, commonNeighbors($"sNeighbors", $"tNeighbors")).otherwise(0.5))
        .withColumn("jaccardCoefficient", when($"sNeighbors".isNotNull && $"tNeighbors".isNotNull, bround(jaccardCoefficient($"sNeighbors", $"tNeighbors"),3)).otherwise(0.01))

      assembler.transform(tempDF4).cache()

    }

    // Read the contents of files in dataframes
    val groundTruthNetworkDF = ss.read
      .option("header", "false")
      .option("sep", "\t")
      .option("inferSchema", "true")
      .csv(groundTruthNetworkFile)
      .toDF("gtsId", "gttId")
      .withColumn("label", lit(1.0))

    val nodeInfoDF = transformNodeInfo(ss.read
      .option("header", "false")
      .option("sep", ",")
      .option("inferSchema", "true")
      .csv(nodeInfoFile)
      .toDF("id", "year", "title", "authors", "journal", "abstract")).cache()

    val trainingSetDF = ss.read
      .option("header", "false")
      .option("sep", " ")
      .option("inferSchema", "true")
      .csv(trainingSetFile)
      .toDF("sId", "tId", "labelTmp")
      .withColumn("label", toDoubleUDF($"labelTmp"))
      .drop("labelTmp")

    //trainingSetDF.show(10)

    val testingSetDF = ss.read
      .option("header", "false")
      .option("sep", " ")
      .option("inferSchema", "true")
      .csv(testingSetFile)
      .toDF("sId", "tId")
      .join(groundTruthNetworkDF, $"sId" === $"gtsId" && $"tId" === $"gttId", "left")
      .drop("gtsId").drop("gttId")
      .withColumn("label", when($"label" === 1.0, $"label").otherwise(0.0))

    val graphDF = transformGraph(
      Graph.fromEdgeTuples(
        trainingSetDF
          .filter($"label" === 1.0)
          .select("sId", "tId")
          .rdd.map(r => (r.getInt(0), r.getInt(1))), 1 // tuples
      )
    )


    val transformedTrainingSetDFPrepro = transformSet(trainingSetDF, nodeInfoDF, graphDF).createOrReplaceTempView("transformedTrainingSetDFPrepro")

    val transformedTrainingSetDF = ss.sql("SELECT DISTINCT(a.*) FROM transformedTrainingSetDFPrepro AS a " +
      "WHERE a.sYear >= a.tYear " )
      .cache() //for performance

    val transformedTestingSetDF = transformSet(testingSetDF, nodeInfoDF, graphDF)
      .cache()

    //transformedTestingSetDF.show(10)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val LRmodel = new LogisticRegression()
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setElasticNetParam(0.0)
      .fit(transformedTrainingSetDF)

    val predictionsLR = LRmodel.transform(transformedTestingSetDF)
    predictionsLR.printSchema()
    predictionsLR.show(10)

    val accuracyLR = evaluator.evaluate(predictionsLR)
    println("F1-score of Logistic Regression: " + accuracyLR)

    val RFModel = new RandomForestClassifier().setNumTrees(15)
      .fit(transformedTrainingSetDF)

    val predictionsRF = RFModel.transform(transformedTestingSetDF)
    predictionsRF.printSchema()
    predictionsRF.show(10)

    val accuracyRF = evaluator.evaluate(predictionsRF)
    println("F1-score of Random Forest: " + accuracyRF)

    println("Importance of features: " + RFModel.featureImportances)

    val DTModel = new DecisionTreeClassifier()
      .setMaxDepth(10)
      .setImpurity("entropy")
      .fit(transformedTrainingSetDF)

    val predictionsDT = DTModel.transform(transformedTestingSetDF)
    //predictionsDT.printSchema()
    predictionsDT.show(10)

    val accuracyDT = evaluator.evaluate(predictionsDT)
    println("F1-score of Decision Tree : " + accuracyDT)


    //    System.in.read()
    ss.stop()
  }
}
