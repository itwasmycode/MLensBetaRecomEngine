import org.apache.spark.SparkConf
import org.apache.log4j._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}

trait MLensWrapper{
  // Get rid of useless Spark Logs(useless for this process not at all.)
  Logger.getLogger("org").setLevel(Level.ERROR)

  // Creating spark cluster configuration at local.
  val conf : SparkConf = new SparkConf()
    .setMaster("local[*]")

  // Creating session that uses our configuration. If there is one session instead of create, use it.
  lazy val spSess: SparkSession = SparkSession
                  .builder
                  .config(conf)
                  .appName("MovieLensRecommendation")
                  .getOrCreate()

  // Creting DataFrame from csv file using "userId","movieId","rating","timestamp"
  val ratingsDF : DataFrame = spSess.read
                  .option("header","true")
                  .csv("src/data/ratings.csv")
                  .toDF("userId","movieId","rating","timestamp")

  // Creating DataFrame from csv file using "movieId","movieName", "genres
  val moviesDF : DataFrame = spSess.read
                  .option("header","TRUE")
                  .csv("src/data/movies.csv")
                  .toDF("movieId","movieName","genres")

  // Casting movies DataFrame movieId column to IntegerType
  val updatedMovieDF : DataFrame = moviesDF
                  .withColumn("movieId",col("movieId").cast(IntegerType))


  // Casting columns to desired types.
  val updatedDF : DataFrame = ratingsDF
                  .withColumn("userId",col("userID").cast(IntegerType))
                  .withColumn("movieId",col("movieId").cast(IntegerType))
                  .withColumn("rating",col("rating").cast(DoubleType))
                  .withColumn("timestamp",col("timestamp").cast(IntegerType))
                  .join(updatedMovieDF,"movieId")
}