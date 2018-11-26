import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, NaiveBayes, RandomForestClassifier}
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

    def dotProductUDF = udf(
      (a: linalg.Vector, b: linalg.Vector) => {
        val aArr = a.toArray
        val bArr = b.toArray
        aArr.zip(bArr).map(t => t._1 * t._2).sum
      }
    )

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

    def transformSet(input: DataFrame, nodeInfo: DataFrame): DataFrame = {
      val assembler = new VectorAssembler()
        .setInputCols(Array("yearDiff", "isSameJournal","cosSimTFIDF"))
        .setOutputCol("features")

      val tempDF = input
        .join(nodeInfo
          .select($"id",
            $"year".as("sYear"),
            $"journal".as("sJournal"),
            $"abstractFeatures".as("sAbstractFeatures")), $"sId" === $"id")
        .drop("id")
        .join(nodeInfo
          .select($"id",
            $"year".as("tYear"),
            $"journal".as("tJournal"),
            $"abstractFeatures".as("tAbstractFeatures")), $"tId" === $"id")
        .drop("id")
        .withColumn("yearDiff", $"sYear" - $"tYear")
        .withColumn("isSameJournal", when($"sJournal" === $"tJournal", true).otherwise(false))
        .withColumn("cosSimTFIDF", dotProductUDF($"sAbstractFeatures", $"tAbstractFeatures"))
        .drop("sAbstractFeatures").drop("tAbstractFeatures")

      assembler.transform(tempDF)
    }

    // Read the contents of files in dataframes
    val groundTruthNetworkDF = ss.read
      .option("header", "false")
      .option("sep", "\t")
      .option("inferSchema", "true")
      .csv(groundTruthNetworkFile)
      .toDF("gtsId", "gttId")
      .withColumn("label", lit(1.0)).cache()

    val nodeInfoDF = transformNodeInfo(ss.read
      .option("header", "false")
      .option("sep", ",")
      .option("inferSchema", "true")
      .csv(nodeInfoFile)
      .toDF("id", "year", "title", "authors", "journal", "abstract"))

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
    ).cache()

    val testingSetDF = transformSet(
      ss.read
        .option("header", "false")
        .option("sep", " ")
        .option("inferSchema", "true")
        .csv(testingSetFile)
        .toDF("sId", "tId")
        .join(groundTruthNetworkDF, $"sId" === $"gtsId" && $"tId" === $"gttId", "left")
        .drop("gtsId").drop("gttId")
        .withColumn("label", when($"label" === 1.0, $"label").otherwise(0.0)),
      nodeInfoDF
    ).cache()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val LRmodel = new LogisticRegression()
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setElasticNetParam(0.0)
      .fit(trainingSetDF)

    val predictionsLR = LRmodel.transform(testingSetDF)
    predictionsLR.printSchema()
    predictionsLR.show(10)

    val accuracyLR = evaluator.evaluate(predictionsLR)
    println("F1-score of Logistic Regression: " + accuracyLR)

    val RFModel = new RandomForestClassifier()
      .fit(trainingSetDF)

    val predictionsRF = RFModel.transform(testingSetDF)
    predictionsRF.printSchema()
    predictionsRF.show(10)

    val accuracyRF = evaluator.evaluate(predictionsRF)
    println("F1-score of Random Forest: " + accuracyRF)

    ss.stop()
  }
}
