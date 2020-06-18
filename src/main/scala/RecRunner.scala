// Importing the required packages.
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.expressions.UserDefinedFunction


object RecRunner extends MLensWrapper {
  def main(args: Array[String]): Unit = {

    // In order to avoid randomness of spread of dataset; shuffling them.
    val shuffledDF = updatedDF.orderBy(rand())

    // Because there is not splitted Genre DataFrame, we split genres [GenreA|GenreB] -> GenreA,GenreB
    // and collect them to make one-hot-encoded.
    val targetColumns : Array[String] = shuffledDF
        .withColumn("genres"
        ,explode(split(col("genres"),"\\|")))
        .dropDuplicates()
        .select("genres")
        .distinct()
        .rdd.map(r => r(0).toString)
        .collect()

    // If item in list return 1 else 0
    val cateListContains : UserDefinedFunction =
      udf((cateList: String, item: String) => if (cateList.contains(item)) 1 else 0)

    // Applying foldLeft function to targetColumns in shuffledDF declared above.
    val splittedDFNew1 : DataFrame = targetColumns.foldLeft(shuffledDF) {
      case (df, item) =>
        df.withColumn(item, cateListContains(col("genres"), lit(item)))
    }

    // Wrong column name, probably one of them doesn't belong to genres in targetCols.
    val finalDFGenres : DataFrame = splittedDFNew1
                      .drop("genres")
                      .drop("(no genres listed)")

    // Extracting movie names using Spark Regex func.
    val finalMovieDF = finalDFGenres
                      .withColumn("movie_year"
                      ,regexp_extract(col("movieName")
                      , "\\(([0-9]\\d+)",1))

    // Making Movie Year String to Int.
    val finalMovieYearCastedDF : DataFrame = finalMovieDF
                        .withColumn("movieYear"
                        ,col("movie_year").cast(IntegerType))
                        .drop("movie_year")

    // In result of experimental design, normalizing rating improves accuracy and other metrics.
    val normalizedDF : DataFrame = finalMovieYearCastedDF
                          .select(mean("rating").alias("mean_rating")
                          ,stddev("rating").alias("stddev_rating"))
                          .crossJoin(finalMovieYearCastedDF)
                          .withColumn("rating_scaled"
                          ,(col("rating") - col("mean_rating")) / col("stddev_rating"))
                          .drop("mean_rating","stddev_rating","rating")

    // Splitting DataFrame Training, Test
    val Array(training, test) = normalizedDF
                              .randomSplit(Array(0.8, 0.2))

    // Caching improves time complexity.
    training.cache()
    test.cache()

    // Creating model based on parameters that gives us best performance. (Optimized, hyparameters)
    val als: ALS = new ALS()
                  .setUserCol("userId")
                  .setItemCol("movieId")
                  .setRatingCol("rating_scaled")
                  .setRank(5)
                  .setRegParam(0.1)
                  .setMaxIter(5)
    // Fitting models
    val model: ALSModel = als.fit(training)

    model.setColdStartStrategy("drop")
    // Measuring accuracy based on test set, in order to avoid overfitting.
    val predictions : DataFrame = model.transform(test).na.drop

    // Since it is continuous problem, make evaluator reference continuous.
    val evaluator : RegressionEvaluator = new RegressionEvaluator()
                  .setLabelCol("rating_scaled")
                  .setPredictionCol("prediction")

    // In case of getting best parameter, we can measure how good is our model.
    def evaluatorFunc(metricParam : String): Unit ={
      evaluator.setMetricName(metricParam).evaluate(predictions)
      for(vl<-Array("rmse","mse","r2","mae"))
        println(s"$vl metric score = "+ evaluatorFunc(vl))
    }

    //Creating model pipeline.
    val pipe = new Pipeline()

    // Creating pipeline array using our ALS model.
    val pipeline1 = Array[PipelineStage](als)

    // Creating grid, based on our parameters
    val pipeline1Grid = new ParamGridBuilder()
                  .baseOn(pipe.stages -> pipeline1)
                  .addGrid(als.maxIter, 1.to(10).by(1))
                  .addGrid(als.regParam, 0.1.to(0.9).by(0.1))
                  .addGrid(als.rank, 5.to(30).by(5))
                  .build()

    // Creating cross validator, using grids, pipelines evaluator func.
    // We use 5 fold CV in this case. Using different part of data gives model that has less error.
    // Splitting and measuring data can be involved random process.
    // My machine has 2 cores, I setted it to 2.
    val cv = new CrossValidator()
                  .setEstimator(pipe)
                  .setEvaluator(evaluator)
                  .setEstimatorParamMaps(pipeline1Grid)
                  .setNumFolds(5)
                  .setParallelism(2)

    // Fitting CV
    val cvModel = cv.fit(training)

    // In order to extract best model from CVModel We declared implicit class.
    implicit class BestParamMapCrossValidatorModel(cvModel: CrossValidatorModel) {
      def bestEstimatorParamMap: ParamMap = {
        cvModel.getEstimatorParamMaps
          .zip(cvModel.avgMetrics)
          .maxBy(_._2)
          ._1
      }
    }

    println(cvModel.bestEstimatorParamMap)

    // Recommending item for given user, DataFrame and how many item we have to suggest.
    def recommendForGiven(userId: Array[Int], modelDF: DataFrame, itemCount: Int):Unit = {
      val toRecUsers :DataFrame = modelDF.select("userId")
        .filter(s"userId==$userId")
        .distinct()

        model.recommendForUserSubset(toRecUsers,itemCount)
    }

    // When we are done, just stop session...
    spSess.stop()
  }
}