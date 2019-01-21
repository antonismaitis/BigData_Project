import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object LinkPredictionPartB {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    System.setProperty("hadoop.home.dir", "C:\\Program Files\\JetBrains\\IntelliJ IDEA 2018.3\\")
    val master = "local[*]"
    val appName = "Link Prediction Part B"
    // Create the spark session first
    val conf: SparkConf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)
      .set("spark.driver.allowMultipleContexts", "false")
      .set("spark.scheduler.mode", "FAIR")
      .set("spark.ui.enabled", "true")

    val ss: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    // Suppress info messages
    ss.sparkContext.setLogLevel("ERROR")

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

    def filterAndEvaluate(input: DataFrame, arr: Array[Double]) = {
      arr.foreach(jaccardSimilarityThreshold => {
        println("*************************************************")
        println("Jaccard Similarity Threshold: ", jaccardSimilarityThreshold)

        val filteredDF = input.filter($"jaccardSimilarity" > jaccardSimilarityThreshold)

        println("Filtered count: ", filteredDF.count())
        println("Correctly labeled: ", filteredDF.filter($"correct" === 1.0).count())
        println("Incorrectly labeled: ", filteredDF.filter($"correct" === 0.0).count())
        println("\n")
      })
    }


    def transformNodeInfo(input: DataFrame): DataFrame = {
      val abstractTokenizer = new RegexTokenizer()
        .setMinTokenLength(3)
        .setInputCol("abstract")
        .setOutputCol("abstractWords")
      val abstractRemover = new StopWordsRemover()
        .setInputCol(abstractTokenizer.getOutputCol)
        .setOutputCol("abstractFilteredWords")
      val abstractHashingTF = new HashingTF()
        .setInputCol(abstractRemover.getOutputCol)
        .setOutputCol("abstractFeatures")
        .setBinary(true)
      val abstractCountVectorizer = new CountVectorizer()
        .setInputCol(abstractRemover.getOutputCol)
        .setOutputCol("abstractFeatures")
        .setBinary(true)

      val titleTokenizer = new RegexTokenizer()
        .setMinTokenLength(3)
        .setInputCol("title")
        .setOutputCol("titleWords")
      val titleRemover = new StopWordsRemover()
        .setInputCol(titleTokenizer.getOutputCol)
        .setOutputCol("titleFilteredWords")
      val titleHashingTF = new HashingTF()
        .setNumFeatures(20000)
        .setInputCol(titleRemover.getOutputCol)
        .setOutputCol("titleFeatures")
        .setBinary(true)
      val titleCountVectorizer = new CountVectorizer()
        .setInputCol(titleRemover.getOutputCol)
        .setOutputCol("titleFeatures")
        .setBinary(true)

      val pipeline = new Pipeline()
        .setStages(
          Array(
            abstractTokenizer, abstractRemover, abstractCountVectorizer,
            titleTokenizer, titleRemover, titleCountVectorizer
          )
        )

      val inputCleanedDF = input
        .na.fill(Map("abstract" -> "", "title" -> ""))

      val model = pipeline.fit(inputCleanedDF)

      model
        .transform(inputCleanedDF)
        .filter($"abstract" =!= "")
        .filter(!vectorEmpty($"abstractFeatures"))
        .filter($"title" =!= "")
        .filter(!vectorEmpty($"titleFeatures"))
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

    //    nodeInfoDF.show(false)


    val mh = new MinHashLSH()
      .setNumHashTables(5)
      .setInputCol("abstractFeatures")
      .setOutputCol("hashes")

    //    val nodeInfoSampleDF = nodeInfoDF.sample(false, 0.1);

    val model = mh.fit(nodeInfoDF)

    val transformedNodeInfoDF = model.transform(nodeInfoDF).cache()

    // Approximate similarity join
    val approxSimJoinDF = model.approxSimilarityJoin(transformedNodeInfoDF, transformedNodeInfoDF, 0.95, "JaccardDistance")
      .select($"datasetA.id".as("idA"),
        $"datasetA.abstractFeatures".as("abstractFeaturesA"),
        $"datasetA.year".as("yearA"),
        $"datasetB.id".as("idB"),
        $"datasetB.abstractFeatures".as("abstractFeaturesB"),
        $"datasetB.year".as("yearB"),
        $"JaccardDistance")
      .filter("idA < idB")

    val transformedDF = approxSimJoinDF
      .withColumn("sId", when($"yearA" > $"yearB", $"idA").otherwise($"idB"))
      .withColumn("tId", when($"yearA" > $"yearB", $"idB").otherwise($"idA"))
      .join(groundTruthNetworkDF, $"sId" === $"gtsId" && $"tId" === $"gttId", "left")
      .drop("gtsId").drop("gttId")
      .withColumn("correct", when($"label" === 1.0, $"label").otherwise(0.0))
      .withColumn("jaccardSimilarity", jaccardSimilarity($"abstractFeaturesA", $"abstractFeaturesB"))
      .drop("abstractFeaturesA").drop("abstractFeaturesB")
      .cache()

    println("Total count: ", transformedDF.count())

    filterAndEvaluate(transformedDF, Array(0.05, 0.1, 0.15, 0.2, 0.25))

    //    val filteredDF = transformedDF.filter($"jaccardSimilarity" > 0.1)
    //
    //    filteredDF.printSchema()
    //    filteredDF.show(false)
    //    println("Filtered count: ",filteredDF.count())
    //    println("Correctly labeled: ",filteredDF.filter($"correct" === 1.0).count())
    //    println("Incorrectly labeled: ",filteredDF.filter($"correct" === 0.0).count())

    System.in.read()
    ss.stop()
  }
}
