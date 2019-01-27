import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object LinkPredictionPartB {
  def main(args: Array[String]): Unit = {
    val t0 = System.nanoTime

    // Create the spark session first
    System.setProperty("hadoop.home.dir", "C:\\Program Files\\JetBrains\\IntelliJ IDEA 2018.3\\")
    val master = "local[*]"
    val appName = "Link Prediction Part B"
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
      //.set("spark.sql.shuffle.partitions","42") // Number of partitions = Total input dataset size / partition size => 1500 / 64 = 23.43 = ~23 partitions.
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

    val nodeInfoFile = "./node_information.csv"
    val groundTruthNetworkFile = "./Cit-HepTh.txt"
    val SEED = 1L
    val NUM_HASH_TABLES = 1
    val JACCARD_DISTANCE_THRESHOLD = 0.85

    def vectorEmpty = udf(
      (vec: linalg.Vector) => vec.numNonzeros == 0
    )

    def filterAndEvaluate(input: DataFrame, arr: Array[Double]): Unit = {
      arr.foreach(jaccardDistanceThreshold => {
        println("*************************************************")
        println("Jaccard Distance Threshold: " + jaccardDistanceThreshold)

        val filteredDF = input.filter($"jaccardDistance" < jaccardDistanceThreshold)

        println("Filtered count: " + filteredDF.count())
        println("Correctly labeled: " + filteredDF.filter($"correct" === 1.0).count())
        println("Incorrectly labeled: " + filteredDF.filter($"correct" === 0.0).count())
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
      //val abstractCountVectorizer = new CountVectorizer()
       // .setInputCol(abstractRemover.getOutputCol)
       // .setOutputCol("abstractFeatures")
       // .setBinary(true)

      val abstractHashingTF = new HashingTF()
        .setInputCol(abstractRemover.getOutputCol)
        .setOutputCol("abstractFeatures")
        .setNumFeatures(18000)
        .setBinary(true)



      val titleTokenizer = new RegexTokenizer()
        .setMinTokenLength(3)
        .setInputCol("title")
        .setOutputCol("titleWords")
      val titleRemover = new StopWordsRemover()
        .setInputCol(titleTokenizer.getOutputCol)
        .setOutputCol("titleFilteredWords")
     // val titleCountVectorizer = new CountVectorizer()
       // .setInputCol(titleRemover.getOutputCol)
        //.setOutputCol("titleFeatures")
        //.setBinary(true)

      val titleHashingTF = new HashingTF()
        .setInputCol(titleRemover.getOutputCol)
        .setOutputCol("titleFeatures")
        .setBinary(true)
        .setNumFeatures(8000)

      val assembler1 = new VectorAssembler()
        .setInputCols(Array("titleFeatures","abstractFeatures"))
        .setOutputCol("ffeatures")







      val pipeline = new Pipeline()
        .setStages(
          Array(
            abstractTokenizer, abstractRemover, abstractHashingTF,
            titleTokenizer, titleRemover, titleHashingTF,assembler1
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


    val mh = new MinHashLSH()
      .setSeed(SEED)
      .setNumHashTables(NUM_HASH_TABLES)
      .setInputCol("ffeatures")
      .setOutputCol("hashes")

    val model = mh.fit(nodeInfoDF)

    val transformedNodeInfoDF = model.transform(nodeInfoDF).select("id", "year", "ffeatures", "hashes").cache()

    // Approximate similarity join
    val approxSimJoinDF = model.approxSimilarityJoin(transformedNodeInfoDF, transformedNodeInfoDF, JACCARD_DISTANCE_THRESHOLD, "JaccardDistance")
      .select($"datasetA.id".as("idA"),
        $"datasetA.year".as("yearA"),
        $"datasetB.id".as("idB"),
        $"datasetB.year".as("yearB"),
        $"JaccardDistance")
      .filter("idA < idB")

    val transformedDF = approxSimJoinDF
      .withColumn("sId", when($"yearA" > $"yearB", $"idA").otherwise($"idB"))
      .withColumn("tId", when($"yearA" > $"yearB", $"idB").otherwise($"idA"))
      //      .drop("idA", "idB", "yearA", "yearB")
      .join(groundTruthNetworkDF, $"sId" === $"gtsId" && $"tId" === $"gttId" , "left")
      .drop("gtsId").drop("gttId")
      .withColumn("correct", when($"label" === 1.0, $"label").otherwise(0.0))
      .drop("ffeaturesA", "ffeaturesB")
      .cache()

    transformedDF.show(false)
    println("Total count: ", transformedDF.count())

    filterAndEvaluate(transformedDF, 0.05.to(JACCARD_DISTANCE_THRESHOLD + 0.0001).by(0.05).toArray)

    println("Elapsed time: " + (System.nanoTime - t0) / 1e9d)

    System.in.read()
    ss.stop()
  }
}