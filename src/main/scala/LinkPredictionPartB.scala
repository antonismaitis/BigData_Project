import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object LinkPredictionPartB {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("linkPrediction")
      .config("spark.sql.shuffle.partitions", 8)
      .getOrCreate()
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    // Suppress info messages
    ss.sparkContext.setLogLevel("ERROR")

    //    ss.sqlContext.setConf("spark.sql.shuffle.partitions", "8")


    val trainingSetFile = "./training_set.txt"
    val testingSetFile = "./testing_set.txt"
    val nodeInfoFile = "./node_information.csv"
    val groundTruthNetworkFile = "./Cit-HepTh.txt"

    def toDoubleUDF = udf(
      (n: Int) => n.toDouble
    )

    def jaccardSimilarity = udf(
      (a: linalg.Vector, b: linalg.Vector) => {
        val aArr = a.toArray
        val bArr = b.toArray
        val zippedArr = aArr.zip(bArr)
        zippedArr.map(t => t._1 * t._2).sum /
          zippedArr.map(t => t._1 + t._2).count(t => t != 0.0).toDouble

      }
    )

    def vectorEmpty = udf(
      (vec: linalg.Vector) => vec.equals(Vectors.zeros(vec.size))
    )

    def transformNodeInfo(input: DataFrame): DataFrame = {
      val tokenizer = new Tokenizer().setInputCol("abstract").setOutputCol("abstractWords")
      val wordsDF = tokenizer.transform(input.na.fill(Map("abstract" -> "")))

      val remover = new StopWordsRemover()
        .setInputCol("abstractWords")
        .setOutputCol("abstractFilteredWords")
      val filteredWordsDF = remover.transform(wordsDF)

      val cvModel: CountVectorizerModel = new CountVectorizer()
        .setInputCol("abstractFilteredWords")
        .setOutputCol("abstractFeatures")
        .setBinary(true)
        .fit(filteredWordsDF)

      val completeDF = cvModel.transform(filteredWordsDF)
      completeDF.filter($"abstract" =!= "")
        .filter(!vectorEmpty($"abstractFeatures"))
    }


    def transformSet(input: DataFrame, nodeInfo: DataFrame): DataFrame = {
      val assembler = new VectorAssembler()
        .setInputCols(Array("jaccardSimilarity"))
        .setOutputCol("features")

      val tempDF = input
        .join(nodeInfo
          .select($"id",
            $"abstractFeatures".as("sAbstractFeatures")), $"sId" === $"id")
        .drop("id")
        .join(nodeInfo
          .select($"id",
            $"abstractFeatures".as("tAbstractFeatures")), $"tId" === $"id")
        .drop("id")
        .withColumn("jaccardSimilarity", jaccardSimilarity($"sAbstractFeatures", $"tAbstractFeatures"))
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
      .withColumn("label", lit(1.0))

    val nodeInfoDF = transformNodeInfo(
      ss.read
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

    val mh = new MinHashLSH()
      //      .setNumHashTables(5)
      .setInputCol("abstractFeatures")
      .setOutputCol("hashes")

    val nodeInfoSampleDF = nodeInfoDF.sample(false, 0.1);

    val model = mh.fit(nodeInfoDF)

    val transformedNodeInfoDF = model.transform(nodeInfoDF).cache()

    // Approximate similarity join
    model.approxSimilarityJoin(transformedNodeInfoDF, transformedNodeInfoDF, 0.6).filter("datasetA.id < datasetB.id").count




    val transformedTrainingSetDF = transformSet(trainingSetDF, nodeInfoDF)
      .cache()
    val transformedTestingSetDF = transformSet(testingSetDF, nodeInfoDF)
      .cache()
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

    System.in.read()
    ss.stop()
  }
}
