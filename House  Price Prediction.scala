// Databricks notebook source
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md ### Load Source Data

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val data = spark.read.option("inferSchema","true").option("header","true").csv("/FileStore/tables/train.csv")
// MAGIC 
// MAGIC display(data)

// COMMAND ----------

data.printSchema()

// COMMAND ----------

// MAGIC %md ### Prepare the Training Data
// MAGIC To train the regression model, you need a training data set that includes a vector of numeric features, and a label column. In this project, you will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **SalePrice** column to **label**.

// COMMAND ----------

// DBTITLE 1,List all String Data Type Columns in an Array in further processing
// MAGIC %scala
// MAGIC 
// MAGIC var StringfeatureCol = Array("MSZoning", "LotFrontage", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature")

// COMMAND ----------

// DBTITLE 1,Example of StringIndexer
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

display(indexed)

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.attribute.Attribute
// MAGIC import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
// MAGIC import org.apache.spark.ml.{Pipeline, PipelineModel}
// MAGIC 
// MAGIC val indexers = StringfeatureCol.map { colName =>
// MAGIC   new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
// MAGIC }
// MAGIC 
// MAGIC val pipeline = new Pipeline()
// MAGIC                     .setStages(indexers)      
// MAGIC 
// MAGIC val HouseDF = pipeline.fit(data).transform(data)

// COMMAND ----------

HouseDF.printSchema()

// COMMAND ----------

HouseDF.show()

// COMMAND ----------

// MAGIC %md ### Split the Data
// MAGIC It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this project, you will use 70% of the data for training, and reserve 30% for testing. In the testing data, the **label** column is renamed to **trueLabel** so you can use it later to compare predicted labels with known actual values.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val splits = HouseDF.randomSplit(Array(0.7, 0.3))
// MAGIC val train = splits(0)
// MAGIC val test = splits(1)
// MAGIC val train_rows = train.count()
// MAGIC val test_rows = test.count()
// MAGIC println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// DBTITLE 1,VectorAssembler() that combines categorical features into a single vector
// MAGIC %scala
// MAGIC 
// MAGIC val assembler = new VectorAssembler().setInputCols(Array("Id", "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath","BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold", "MSZoning_indexed", "LotFrontage_indexed", "Street_indexed", "Alley_indexed", "LotShape_indexed","LandContour_indexed", "Utilities_indexed", "LotConfig_indexed", "LandSlope_indexed", "Neighborhood_indexed", "Condition1_indexed", "Condition2_indexed", "BldgType_indexed", "HouseStyle_indexed", "RoofStyle_indexed", "RoofMatl_indexed", "Exterior1st_indexed", "Exterior2nd_indexed", "MasVnrType_indexed", "MasVnrArea_indexed", "ExterQual_indexed", "ExterCond_indexed", "Foundation_indexed", "BsmtQual_indexed", "BsmtCond_indexed", "BsmtExposure_indexed", "BsmtFinType1_indexed", "BsmtFinType2_indexed", "Heating_indexed", "HeatingQC_indexed", "CentralAir_indexed", "Electrical_indexed", "KitchenQual_indexed", "Functional_indexed", "FireplaceQu_indexed", "GarageType_indexed", "GarageYrBlt_indexed", "GarageFinish_indexed", "GarageQual_indexed", "GarageCond_indexed", "PavedDrive_indexed", "PoolQC_indexed", "Fence_indexed", "MiscFeature_indexed" )).setOutputCol("features")
// MAGIC val training = assembler.transform(train).select($"features", $"SalePrice".alias("label"))
// MAGIC training.show()

// COMMAND ----------

// MAGIC %md ### Train a Regression Model
// MAGIC Next, you need to train a regression model using the training data. To do this, create an instance of the regression algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this Project, you will use a *Linear Regression* algorithm - though you can use the same technique for any of the regression algorithms supported in the spark.ml API.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
// MAGIC val model = lr.fit(training)
// MAGIC println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Prepare the Testing Data
// MAGIC Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **SalePrice** column to **trueLabel**.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val testing = assembler.transform(test).select($"features", $"SalePrice".alias("trueLabel"))
// MAGIC testing.show()

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. But in this case you are using the test data which includes a known true label value, so you can compare the predicted Sale Price. 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val prediction = model.transform(testing)
// MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
// MAGIC predicted.show()

// COMMAND ----------

// MAGIC %md Looking at the result, the **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data. It looks like there is some variance between the predictions and the actual values (the individual differences are referred to as *residuals*) you'll learn how to measure the accuracy of a model.

// COMMAND ----------

// MAGIC %md ## Evaluating a Regression Model
// MAGIC 
// MAGIC In this Project, we have created pipeline for a linear regression model, and then test and evaluate the model.
// MAGIC 
// MAGIC ### Prepare the Data
// MAGIC 
// MAGIC First, import the libraries you will need and prepare the training and test data:

// COMMAND ----------

// MAGIC %md ### Examine the Predicted and Actual Values
// MAGIC You can plot the predicted values against the actual values to see how accurately the model has predicted. In a perfect model, the resulting scatter plot should form a perfect diagonal line with each predicted value being identical to the actual value - in practice, some variance is to be expected.
// MAGIC Run the cells below to create a temporary table from the **predicted** DataFrame and then retrieve the predicted and actual label values using SQL. You can then display the results as a scatter plot, specifying **-** as the function to show the unaggregated values.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC predicted.createOrReplaceTempView("HousePrice")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select prediction, trueLabel from HousePrice

// COMMAND ----------

// MAGIC %md ### Retrieve the Root Mean Square Error (RMSE)
// MAGIC There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the predicted and actual values - so in this case, the RMSE indicates the average number of minutes between predicted and actual Sale Price values. You can use the **RegressionEvaluator** class to retrieve the RMSE.

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluator = new RegressionEvaluator().setLabelCol("trueLabel").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(prediction)
println("Root Mean Square Error (RMSE): " + (rmse))
