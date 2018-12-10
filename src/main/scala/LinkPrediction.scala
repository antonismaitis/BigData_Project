import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.io.Source // from the Scala standard library
import java.io.{PrintWriter, File, FileOutputStream}
import au.com.bytecode.opencsv.CSVWriter
import java.io.BufferedWriter
import java.io.FileWriter
import scala.collection.JavaConverters._
import scala.collection.JavaConverters._
import org.apache.spark.sql.functions.col

class AsArrayList[T](input: List[T]) {
  def asArrayList: java.util.ArrayList[T] = new java.util.ArrayList[T](input.asJava)
}

object LinkPrediction {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate;
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

    def transformSet(input: DataFrame, nodeInfo: DataFrame): DataFrame = {
      val assembler = new VectorAssembler()
        .setInputCols(Array("yearDiff"))
        .setOutputCol("features")

      val tempDF = input
        .join(nodeInfo.select("id", "year"), $"sId" === $"id")
        .withColumnRenamed("year", "sYear")
        .drop("id")
        .join(nodeInfo.select("id", "year"), $"tId" === $"id")
        .withColumnRenamed("year", "tYear")
        .drop("id")
        .withColumn("yearDiff", abs($"sYear" - $"tYear"))

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

    val nodeInfoDF = ss.read
      .option("header", "false")
      .option("sep", ",")
      .option("inferSchema", "true")
      .csv(nodeInfoFile)
      .toDF("id", "year", "title", "authors", "journal", "abstract")

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

    val trainingSetDF = transformSet(
      trainingSet
        .toDF(colNames: _*)
        .withColumn("label", toDoubleUDF($"labelTmp"))
        .drop("labelTmp","_c0"),
      nodeInfoDF
    )
//    val cond = trainingSetDF.columns.map(x => col(x).isNull || col(x) === "").reduce(_ || _)
//    df.filter(cond).show
    trainingSetDF.show(10)

    val testingSetDF = transformSet(
      ss.read
        .option("header", "false")
        .option("sep", " ")
        .option("inferSchema", "true")
        .csv(testingSetFile)
        .toDF("sId", "tId")
        .join(groundTruthNetworkDF, $"sId" === $"gtsId" && $"tId" === $"gttId", "left")
        .drop("gtsId").drop("gttId")
        .withColumn("label", when($"label" === 1.0, $"label").otherwise(0.0))
        .withColumn("randomPrediction", when(randn(0) > 0.5, 1.0).otherwise(0.0)),
      nodeInfoDF
    )
    testingSetDF.show(10)

    //
    //
    //    val randomAccuracy = testingSetDF.filter($"label" === $"randomPrediction").count /
    //      testingSetDF.count.toDouble
    //    println(s"Random accuracy: ${randomAccuracy}")
    //
    //
    //    val NBmodel = new NaiveBayes().fit(trainingSetDF)
    //
    //    val predictionsNB = NBmodel.transform(testingSetDF)
    //    predictionsNB.printSchema()
    //    //predictionsNB.take(100).foreach(println)
    //    //predictionsNB.select("label", "prediction").show(100)
    //    predictionsNB.show(10)
    //
    //    // Evaluate the model by finding the accuracy
    //    val evaluatorNB = new MulticlassClassificationEvaluator()
    //      .setLabelCol("label")
    //      .setPredictionCol("prediction")
    //      .setMetricName("accuracy")
    //
    //    val accuracyNB = evaluatorNB.evaluate(predictionsNB)
    //    println("Accuracy of Naive Bayes: " + accuracyNB)
    //
    //    val LRmodel = new LogisticRegression()
    //      .setMaxIter(10000)
    //      .setRegParam(0.1)
    //      .setElasticNetParam(0.0)
    //      .fit(trainingSetDF)
    //
    //    val predictionsLR = LRmodel.transform(testingSetDF)
    //    predictionsLR.printSchema()
    //    predictionsLR.show(10)
    //
    //    val accuracyLR = evaluatorNB.evaluate(predictionsLR)
    //    println("Accuracy of Logistic Regression: " + accuracyLR)

  }
}
