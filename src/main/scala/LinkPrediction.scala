import org.apache.spark.graphx._
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.{ClusteringEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, SparkSession}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel}

import scala.collection.mutable

//import org.apache.spark.ml.feature.MinHashLSH


object LinkPrediction {
  def main(args: Array[String]): Unit = {
    val t0 = System.nanoTime

    System.setProperty("hadoop.home.dir", "C:\\Program Files\\JetBrains\\IntelliJ IDEA 2018.3\\")
    val master = "local[*]"
    val appName = "Link Prediction"
    // Create the spark session first
    val conf: SparkConf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)
      .set("spark.ui.enabled", "true")
      .set("spark.driver.allowMultipleContexts", "false")
      .set("spark.scheduler.mode", "FAIR")
      .set("spark.scheduler.allocation.file","src/resources/fairscheduler.xml")
      .set("spark.scheduler.pool","default")
      .set("spark.memory.fraction","1")
      .set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
      .set("spark.default.parallelism","100")
      .set("spark.sql.shuffle.partitions","42")
      .set("spark.task.cpus","1")
      .set("spark.dynamicAllocation.enabled","true")
      .set("spark.dynamicAllocation.minExecutors","1")
      .set("spark.dynamicAllocation.executorAllocationRatio","1")
      .set("spark.streaming.backPressure.enabled","true")
      .set("spark.streaming.blockInterval","250ms")

    val ss: SparkSession = SparkSession.builder().config(conf).getOrCreate()


    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    // Suppress info messages
    ss.sparkContext.setLogLevel("ERROR")

    val currentDir = System.getProperty("user.dir") // get the current directory
    val trainingSetFile = "./training_set.txt"
    val testingSetFile = "./testing_set.txt"
    val nodeInfoFile = "./node_information.csv"
    val groundTruthNetworkFile = "./Cit-HepTh.txt"

    val SEED = 1L

    def toDoubleUDF = udf(
      (n: Int) => n.toDouble
    )

    def myUDF: UserDefinedFunction = udf(
      (s1: String, s2: String) => {
        val splitted1 = s1.split(",")
        val splitted2 = s2.split(",")
        splitted1.intersect(splitted2).length
      })

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
      val abstractTokenizer = new RegexTokenizer().setMinTokenLength(3).setInputCol("abstract").setOutputCol("abstractWords")
      val abstractRemover = new StopWordsRemover().setInputCol(abstractTokenizer.getOutputCol).setOutputCol("abstractFilteredWords")
      val abstractHashingTF = new HashingTF().setInputCol(abstractRemover.getOutputCol).setOutputCol("abstractRawFeatures").setNumFeatures(20000)
      val abstractIdf = new IDF().setInputCol(abstractHashingTF.getOutputCol).setOutputCol("abstractFeatures")
      val titleTokenizer = new RegexTokenizer().setMinTokenLength(3).setInputCol("title").setOutputCol("titleWords")
      val titleRemover = new StopWordsRemover().setInputCol(titleTokenizer.getOutputCol).setOutputCol("titleFilteredWords")
      val titleHashingTF = new HashingTF().setInputCol(titleRemover.getOutputCol).setOutputCol("titleRawFeatures").setNumFeatures(20000)
      val titleIdf = new IDF().setInputCol(titleHashingTF.getOutputCol).setOutputCol("titleFeatures")
      val numOfClusters = 6
      val kMeans = new KMeans().setK(numOfClusters).setFeaturesCol(abstractIdf.getOutputCol).setSeed(SEED).setPredictionCol("cluster")
      val pipeline = new Pipeline().setStages(Array(
//        abstractTokenizer, titleTokenizer, abstractRemover, titleRemover, abstractHashingTF, titleHashingTF, abstractIdf, titleIdf, kMeans
          abstractTokenizer, abstractRemover, abstractHashingTF, abstractIdf, kMeans
      ))

      val inputCleanedDF = input.na.fill(Map("abstract" -> "", "title" -> ""))
      val model = pipeline.fit(inputCleanedDF)

      model.transform(inputCleanedDF)
    }

    def transformSet(input: DataFrame, nodeInfo: DataFrame, graph: DataFrame): DataFrame = {

      val tempDF = input
        .join(nodeInfo
          .select($"id",
            $"authors".as("sAuthors"),
            $"year".as("sYear"),
            $"journal".as("sJournal"),
            $"cluster".as("sCluster"),
            $"abstractFeatures".as("sAbstractFeatures")//,
//            $"titleFeatures".as("sTitleFeatures")
          ), $"sId" === $"id")
        .drop("id")
        .join(nodeInfo
          .select($"id",
            $"authors".as("tAuthors"),
            $"year".as("tYear"),
            $"journal".as("tJournal"),
            $"cluster".as("tCluster"),
            $"abstractFeatures".as("tAbstractFeatures")//,
//            $"titleFeatures".as("tTitleFeatures")
          ), $"tId" === $"id")
        .drop("id")
        .withColumn("yearDiff", $"tYear" - $"sYear")
        .withColumn("nCommonAuthors", when($"sAuthors".isNotNull && $"tAuthors".isNotNull, myUDF('sAuthors, 'tAuthors)).otherwise(0))
        .withColumn("isSelfCitation", $"nCommonAuthors" >= 1)
        .withColumn("isSameJournal", when($"sJournal" === $"tJournal", true).otherwise(false))
        .withColumn("InSameCluster", when($"sCluster" === $"tCluster", true).otherwise(false))
        .withColumn("cosSimAbstract", cosineSimilarityUDF($"sAbstractFeatures", $"tAbstractFeatures"))
        .drop("sAbstractFeatures").drop("tAbstractFeatures")
//        .withColumn("cosSimTitle", cosineSimilarityUDF($"sTitleFeatures", $"tTitleFeatures"))
//        .drop("sTitleFeatures").drop("tTitleFeatures")
        .join(graph
          .select($"id",
            $"inDegrees".as("sInDegrees"),
            $"neighbors".as("sNeighbors")), $"sId" === $"id", "left")
        .na.fill(Map("sInDegrees" -> 0))
        .drop("id")
        .join(graph
          .select($"id",
            $"inDegrees".as("tInDegrees"),
            $"neighbors".as("tNeighbors")), $"tId" === $"id", "left")
        .na.fill(Map("tInDegrees" -> 0))
        .drop("id")
        .withColumn("inDegreesDiff", $"tInDegrees" - $"sInDegrees")
        .withColumn("commonNeighbors", when($"sNeighbors".isNotNull && $"tNeighbors".isNotNull, commonNeighbors($"sNeighbors", $"tNeighbors")).otherwise(0))
        .withColumn("jaccardCoefficient", when($"sNeighbors".isNotNull && $"tNeighbors".isNotNull, jaccardCoefficient($"sNeighbors", $"tNeighbors")).otherwise(0))

        tempDF
//      assembler.transform(tempDF)
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

    val transformedTrainingSetDF = transformSet(trainingSetDF, nodeInfoDF, graphDF)
      .cache() //for performance
    transformedTrainingSetDF.show(10)


    val transformedTestingSetDF = transformSet(testingSetDF, nodeInfoDF, graphDF)
      .cache()


    def evaluate(trainingSetDF:DataFrame, testingSetDF:DataFrame, features: Array[String]): Unit = {
      if (features.length > 0) {
        val assembler = new VectorAssembler()
          .setInputCols(features)
          .setOutputCol("features")

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val completeTrainingSetDF = assembler.transform(trainingSetDF).cache()
        val completeTestingSetDF = assembler.transform(testingSetDF).cache()


        val LRmodel = new LogisticRegression()
          .setMaxIter(10000)
          .setRegParam(0.1)
          .setElasticNetParam(0.0)
          .fit(completeTrainingSetDF)


        val predictionsLR = LRmodel.transform(completeTestingSetDF)

        println("\n*******************************************************************")
        println("F1-score of Logistic Regression: " + evaluator.evaluate(predictionsLR))

        val RFModel = new RandomForestClassifier()
          .setSeed(SEED)
          .setNumTrees(10)
          .fit(completeTrainingSetDF)

        val predictionsRF = RFModel.transform(completeTestingSetDF)

        println("F1-score of Random Forest: " + evaluator.evaluate(predictionsRF))

        val DTModel = new DecisionTreeClassifier()
          .setSeed(SEED)
          .setMaxDepth(10)
          .setImpurity("entropy")
          .fit(completeTrainingSetDF)

        val predictionsDT = DTModel.transform(completeTestingSetDF)

        println("F1-score of Decision Tree : " + evaluator.evaluate(predictionsDT))
        println("Features: " + features.mkString(", "))
        val featuresImportancesArr = RFModel.featureImportances.toArray
        println("Importance of features: " + featuresImportancesArr.mkString(", "))
        val leastImportantFeature = features(featuresImportancesArr.indexOf(featuresImportancesArr.min))
        println("Feature dropped: " + leastImportantFeature)

        println("*******************************************************************")
        completeTrainingSetDF.unpersist()
        completeTestingSetDF.unpersist()

        evaluate(trainingSetDF, testingSetDF, features.zipWithIndex.filter(_._1 != leastImportantFeature).map(_._1))
      }

    }

    evaluate(transformedTrainingSetDF, transformedTestingSetDF, Array("yearDiff", "nCommonAuthors", "isSelfCitation", "isSameJournal", "cosSimAbstract", "tInDegrees", "inDegreesDiff", "commonNeighbors", "jaccardCoefficient", "InSameCluster"))

    println("Elapsed time: " + (System.nanoTime - t0) / 1e9d)

    System.in.read()
    ss.stop()
  }
}
