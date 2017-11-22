import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{Binarizer, RFormula}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.hadoop.fs._
import java.io._
import java.net.URI
import org.apache.hadoop.mapred.JobConf

object MushroomModel {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("MushroomModel")
    val sc = new SparkContext(conf)

    try {
      val spark = SparkSession.builder().appName("MushroomModel").getOrCreate()

      val mushroomSchema = new StructType(Array (
        new StructField("edible", StringType, true),
        new StructField("capShape", StringType, true),
        new StructField("capSurface", StringType, true),
        new StructField("capColor", StringType, true),
        new StructField("bruises", StringType, true),
        new StructField("odor", StringType, true),
        new StructField("gillAttachment", StringType, true),
        new StructField("gillSpacing", StringType, true),
        new StructField("gillSize", StringType, true),
        new StructField("gillColor", StringType, true),
        new StructField("stalkShape", StringType, true),
        new StructField("stalkRoot", StringType, true),
        new StructField("stalkSurfaceAboveRing", StringType, true),
        new StructField("stalkSurfaceBelowRing", StringType, true),
        new StructField("stalkColorAboveRing", StringType, true),
        new StructField("stalkColorBelowRing", StringType, true),
        new StructField("veilType", StringType, true),
        new StructField("veilColor", StringType, true),
        new StructField("ringNumber", StringType, true),
        new StructField("ringType", StringType, true),
        new StructField("sporePrintColor", StringType, true),
        new StructField("population", StringType, true),
        new StructField("habitat", StringType, true))
      )

      val mushroomData = spark.read.format("csv").schema(mushroomSchema).option("mode", "DROPMALFORMED")
        .load(args(0)+"mushrooms.csv")

      // Referenced this http://www.mushroom-appreciation.com/identify-poisonous-mushrooms.html#sthash.5fuv6iTP.rrx2yecD.dpbs
      createModel("edible ~ capSurface + capShape + stalkColorBelowRing + gillSize + gillColor", mushroomData)

      def createModel(modelFormula : String, df : DataFrame): Unit = {
        // == Logistics Regression Part 1 : Cleaning & Splitting the data
        val supervised = new RFormula().setFormula(modelFormula)
        val fittedLr = supervised.fit(df)
        val preparedDF = fittedLr.transform(df)
        val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))

        val randForest = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features")
        val rfModel = randForest.fit(train)
        rfModel.save(args(1)+"/mushroomModel/")
        val predictions = rfModel.transform(test)

        // == Logistics Regression Part 3: Evaluating the model
        val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
        val rmse = evaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)

        val binarizer: Binarizer = new Binarizer().setInputCol("prediction").setOutputCol("binarized_prediction")
        val predictionBinary = binarizer.transform(predictions)

        // == wrong predictions
        import org.apache.spark.sql.functions._
        val wrongPredictions = predictionBinary.where(expr("label != binarized_prediction"))
        val countErrors = wrongPredictions.groupBy("label").agg(count("prediction").alias("Wrong Predictions"))

        // ==  correct predictions
        val correctPredictions = predictionBinary.where(expr("label == binarized_prediction"))
        val countCorrectedPredictions = correctPredictions.groupBy("label").agg(count("prediction").alias("Correct Predictions"))

        val combinedTable = countErrors.join(countCorrectedPredictions, "label").show()

        var falseNeg = 0.0000
        var falsePos = 0.0000
        var trueNeg = 0.0000
        var truePos = 0.0000

        if(countErrors.select(col("Wrong Predictions")).where((expr("label == 1.0"))).withColumnRenamed("Wrong Predictions", "False Neg").count() > 0){
          falseNeg = countErrors.select(col("Wrong Predictions")).where((expr("label == 1.0"))).withColumnRenamed("Wrong Predictions", "False Neg").first.getLong(0)
        }

        if(countErrors.select(col("Wrong Predictions")).where((expr("label == 0.0"))).withColumnRenamed("Wrong Predictions", "False Pos").count() > 0){
          falsePos = countErrors.select(col("Wrong Predictions")).where((expr("label == 0.0"))).withColumnRenamed("Wrong Predictions", "False Pos").first.getLong(0)
        }

        if(countCorrectedPredictions.select(col("Correct Predictions")).where((expr("label == 0.0"))).withColumnRenamed("Correct Predictions", "True Neg").count() > 0){
          trueNeg =  countCorrectedPredictions.select(col("Correct Predictions")).where((expr("label == 0.0"))).withColumnRenamed("Correct Predictions", "True Neg").first.getLong(0)
        }

        if(countCorrectedPredictions.select(col("Correct Predictions")).where((expr("label == 1.0"))).withColumnRenamed("Correct Predictions", "True Pos").count() > 0){
          truePos = countCorrectedPredictions.select(col("Correct Predictions")).where((expr("label == 1.0"))).withColumnRenamed("Correct Predictions", "True Pos").first.getLong(0)
        }

        val totalN = falseNeg + falsePos + trueNeg + truePos
        val accuracy = ((truePos + trueNeg) / totalN.toDouble)
        val missClassify = (falsePos + falseNeg) / totalN.toDouble
        val truePosRate = truePos / (truePos + falseNeg).toDouble
        val falsePosRate = falsePos / (trueNeg + falsePos).toDouble
        val falseNegRate = falseNeg / (falseNeg + truePos).toDouble

        val fileSystem = FileSystem.get(URI.create(args(1)),new JobConf(MushroomModel.getClass));
        val fsDataOutputStream = fileSystem.create(new Path(args(1)+"mushroomConfusion/mushroom_confusion_matrix.txt"));

        val pw  = new PrintWriter(fsDataOutputStream);
        pw.write("-------------------------------------------------------------------------\n")
        pw.write("| n= " + totalN + "        | Predicted In-Edible | Predicted Edible |    Total   |\n")
        pw.write("-------------------------------------------------------------------------\n")
        pw.write("| Actual In-Edible | TN = " + trueNeg + "            | " + " FP = " + falsePos + "        | " + (trueNeg + falsePos)  + "  |\n")
        pw.write("-------------------------------------------------------------------------\n")
        pw.write("| Actual Edible    | FN = " + falseNeg + "            | " + " TP = " + truePos + "        | " + (falseNeg + truePos) + "  |\n")
        pw.write("-------------------------------------------------------------------------\n")
        pw.write("| Total            | " + (trueNeg + falseNeg) + "                 |  " + (falsePos + truePos) + "             |\n")
        pw.write("-------------------------------------------------------\n")
        pw.write("\nThe accuracy is " + accuracy)
        pw.close();
        fsDataOutputStream.close();
      }

    }
    finally {
      sc.stop()
    }

  }
}
