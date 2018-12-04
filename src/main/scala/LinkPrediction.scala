import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, NaiveBayes, LinearSVC}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._

object LinkPrediction {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("linkPrediction").getOrCreate()
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

    val trainingSetDF = transformSet(
      ss.read
        .option("header", "false")
        .option("sep", " ")
        .option("inferSchema", "true")
        .csv(trainingSetFile)
        .toDF("sId", "tId", "labelTmp")
        .withColumn("label", toDoubleUDF($"labelTmp"))
        .drop("labelTmp"),
      nodeInfoDF
    )


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


    val randomAccuracy = testingSetDF.filter($"label" === $"randomPrediction").count /
      testingSetDF.count.toDouble
    println(s"Random accuracy: ${randomAccuracy}")


    val NBmodel = new NaiveBayes().fit(trainingSetDF)

    val predictionsNB = NBmodel.transform(testingSetDF)
    predictionsNB.printSchema()
    //predictionsNB.take(100).foreach(println)
    //predictionsNB.select("label", "prediction").show(100)
    predictionsNB.show(10)

    // Evaluate the model by finding the accuracy
    val evaluatorNB = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracyNB = evaluatorNB.evaluate(predictionsNB)
    println("Accuracy of Naive Bayes: " + accuracyNB)

    val LRmodel = new LogisticRegression()
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setElasticNetParam(0.0)
      .fit(trainingSetDF)

    val predictionsLR = LRmodel.transform(testingSetDF)
    predictionsLR.printSchema()
    predictionsLR.show(10)

    val accuracyLR = evaluatorNB.evaluate(predictionsLR)
    println("Accuracy of Logistic Regression: " + accuracyLR)

  }
}
