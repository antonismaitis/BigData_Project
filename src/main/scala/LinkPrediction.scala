import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, NaiveBayes, DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.io.Source // from the Scala standard library
import java.io.{PrintWriter, File, FileOutputStream}
import au.com.bytecode.opencsv.CSVWriter
import java.io.BufferedWriter
import java.io.FileWriter
import scala.collection.JavaConverters._
import org.apache.spark._
import org.apache.spark.graphx._
import scala.collection.mutable
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.sql.{Dataset, Encoder, Encoders}
import org.apache.spark.sql.expressions.UserDefinedFunction
//Comment out after first compile
case class MyCase(sId: Int, tId:Int, label:Double, sAuthors:String, sYear:Int, sJournal:String,tAuthors:String, tYear:Int,tJournal:String, yearDiff:Int,nCommonAuthors:Int,isSelfCitation:Boolean
                  ,isSameJournal:Boolean,cosSimTFIDF:Double,sInDegrees:Int,sNeighbors:Array[Long],tInDegrees:Int,tNeighbors:Array[Long],inDegreesDiff:Int,commonNeighbors:Int,jaccardCoefficient:Double)





class AsArrayList[T](input: List[T]) {
  def asArrayList: java.util.ArrayList[T] = new java.util.ArrayList[T](input.asJava)
}

object LinkPrediction {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("linkPrediction").getOrCreate()
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    // Suppress info messages
    ss.sparkContext.setLogLevel("ERROR")

    val currentDir = System.getProperty("user.dir") // get the current directory
    val trainingSetFile = "./training_set.txt"
    val trainingSetFixedFilePath = "./training_set_fixed.csv"
    val testingSetFile = "./testing_set.txt"
    val nodeInfoFile = "./node_information.csv"
    val groundTruthNetworkFile = "./Cit-HepTh.txt"
    val trainingSetFixedFile = new BufferedWriter(new FileWriter("./training_set_fixed.csv"))
    val writer = new CSVWriter(trainingSetFixedFile)

    implicit def asArrayList[T](input: List[T]) = new AsArrayList[T](input)

    def toDoubleUDF = udf(
      (n: Int) => n.toDouble
    )

    def myUDF: UserDefinedFunction = udf(
      (s1: String, s2: String) => {
        val splitted1 = s1.split(",")
        val splitted2 = s2.split(",")
        splitted1.intersect(splitted2).length
      })


    def dotProductUDF = udf(
      (a: linalg.Vector, b: linalg.Vector) => {
        val aArr = a.toArray
        val bArr = b.toArray
        aArr.zip(bArr).map(t => t._1 * t._2).sum
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
      val tokenizer = new Tokenizer().setInputCol("abstract").setOutputCol("abstractWords")
      val wordsDF = tokenizer.transform(input.na.fill(Map("abstract" -> "")))

      val hashingTF = new HashingTF().setInputCol("abstractWords").setOutputCol("abstractRawFeatures").setNumFeatures(20000)
      val featurizedDF = hashingTF.transform(wordsDF)

      val idf = new IDF().setInputCol("abstractRawFeatures").setOutputCol("abstractFeatures")
      val idfM = idf.fit(featurizedDF)
      val completeDF = idfM.transform(featurizedDF)
      completeDF
    }

    def transformSet(input: DataFrame, nodeInfo: DataFrame, graph: DataFrame): DataFrame = {
      val assembler = new VectorAssembler()
        //        .setInputCols(Array("yearDiff", "isSameJournal","cosSimTFIDF"))
        .setInputCols(Array("yearDiff","nCommonAuthors","isSelfCitation","isSameJournal", "cosSimTFIDF", "tInDegrees", "inDegreesDiff", "commonNeighbors", "jaccardCoefficient"))
        .setOutputCol("features")

      val tempDF = input
        .join(nodeInfo
          .select($"id",
            $"authors".as("sAuthors"),
            $"year".as("sYear"),
            $"journal".as("sJournal"),
            $"abstractFeatures".as("sAbstractFeatures")), $"sId" === $"id")
        .drop("id")
        .join(nodeInfo
          .select($"id",
            $"authors".as("tAuthors"),
            $"year".as("tYear"),
            $"journal".as("tJournal"),
            $"abstractFeatures".as("tAbstractFeatures")), $"tId" === $"id")
        .drop("id")
        .withColumn("yearDiff", $"tYear" - $"sYear")
        .withColumn("nCommonAuthors", when($"sAuthors".isNotNull && $"tAuthors".isNotNull, myUDF('sAuthors,'tAuthors)).otherwise(0))
        .withColumn("isSelfCitation", $"nCommonAuthors" >=1 )
        .withColumn("isSameJournal", when($"sJournal" === $"tJournal", true).otherwise(false))
        .withColumn("cosSimTFIDF", dotProductUDF($"sAbstractFeatures", $"tAbstractFeatures"))
        .drop("sAbstractFeatures").drop("tAbstractFeatures")
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
        .filter($"sYear" > $"tYear")

      assembler.transform(tempDF)
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

    //read txt and convert it to array
    def readtxtToArray(): java.util.List[Array[String]] = {
      ((Source.fromFile(trainingSetFile)
        .getLines()
        .map(_.split(" ").map(_.trim.toString))).toList).asArrayList
    }


    //fix the values of years looking future years


    //convert it to csv

    writer.writeAll(readtxtToArray())
    trainingSetFixedFile.close()

    val trainingSet = ss.read
      .csv(trainingSetFixedFilePath)

    val colNames = Seq("sId", "tId", "labelTmp")

    val trainingSetDF = ss.read
      .option("header", "false")
      .option("sep", " ")
      .option("inferSchema", "true")
      .csv(trainingSetFile)
      .toDF("sId", "tId", "labelTmp")
      .withColumn("label", toDoubleUDF($"labelTmp"))
      .drop("labelTmp")

    trainingSetDF.show(10)

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
          .rdd.map(r => (r.getInt(0), r.getInt(1))), 1
      )
    )


    val transformedTrainingSetDF = transformSet(trainingSetDF, nodeInfoDF, graphDF)
      .cache()
    val transformedTestingSetDF = transformSet(testingSetDF, nodeInfoDF, graphDF)
    transformedTestingSetDF.show(10)

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

    val RFModel = new RandomForestClassifier()
      .fit(transformedTrainingSetDF)

    val predictionsRF = RFModel.transform(transformedTestingSetDF)
    predictionsRF.printSchema()
    predictionsRF.show(10)

    val accuracyRF = evaluator.evaluate(predictionsRF)
    println("F1-score of Random Forest: " + accuracyRF)

    println("Importance of features: " + RFModel.featureImportances)

    val DTModel = new DecisionTreeClassifier()
      .setMaxDepth(30)
      .setImpurity("entropy")
      .fit(transformedTrainingSetDF)

    val predictionsDT = DTModel.transform(transformedTestingSetDF)
    predictionsDT.printSchema()
    predictionsDT.show(10)

    val accuracyDT = evaluator.evaluate(predictionsDT)
    println("F1-score of Decision Tree : "+accuracyDT)

    val numOfClusters = 2


    val men = Encoders.product[MyCase]

    val ds: Dataset[MyCase] = transformedTrainingSetDF.as(men)


    //KMEANS
    val kmeans = new KMeans().setK(numOfClusters).setSeed(1L)
    val model = kmeans.fit(ds)
    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(ds)
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)


    //BisectingKMEANS
    val bkm = new BisectingKMeans().setK(numOfClusters).setSeed(1L)
    val modelB = bkm.fit(ds)
    // Evaluate clustering.
    val cost = modelB.computeCost(ds)
    println(s"Within Set Sum of Squared Errors = $cost")
    // Shows the result.
    println("Cluster Centers: ")
    val centers = modelB.clusterCenters
    centers.foreach(println)



//    System.in.read()
    ss.stop()
  }
}
