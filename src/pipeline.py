import os
import sys
import traceback
from typing import List

import matplotlib.pyplot as plt
import yaml
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (ClusteringEvaluator,
                                   MulticlassClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.functions import array_to_vector
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, SparkSession

SHOW_LOG = True


class SparkPipeline:
    def __init__(self, config_path: str):
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except:
            print("Unable to load config")
            sys.exit(1)
        try:
            mongo_input_string = f"mongodb://{self.config['mongo']['host']}/{self.config['mongo']['db']}.{self.config['mongo']['collection']}"

            self.spark = (
                SparkSession.builder.master(self.config["spark"]["master"])
                .appName(self.config["spark"]["app_name"])
                .config("spark.mongodb.input.uri", mongo_input_string)
                .config("spark.jars.packages", self.config["mongo"]["package"])
                .getOrCreate()
            )
        except:
            traceback.format_exc()
            print("Unable to create Spark Session. Check configuration file")
            sys.exit(1)
        self.test_size = self.config["test_size"]
        self.n_samples = self.config["n_samples"]
        self.random_seed = self.config["random_seed"]
        self.plot_path = self.config["plot_path"]
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)
        print("Pipeline initialized.")

    def load_data(self):
        try:
            data = (
                self.spark.read.format("mongo")
                .load()
                .limit(self.n_samples)
                .withColumnRenamed("pred", "cluster")
                .drop(*self.config["mongo"]["drop_columns"])
            )
            self.data = data.withColumn("features", array_to_vector(data.features))
            return True
        except:
            print(traceback.format_exc())
            return False

    def split_data(self) -> None:
        try:
            self.train, self.test = self.data.randomSplit(
                weights=[1 - self.test_size, self.test_size], seed=self.random_seed
            )
        except:
            print("Unable to split data")
            sys.exit(1)

    @staticmethod
    def save_training_history(array: List[float], title, path: str) -> None:
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Objective")
        plt.grid(True)
        plt.plot(array)
        plt.savefig(path)

    def eda(self):
        print("Data scheme:")
        self.data.printSchema()
        print("Cluster distribution:")
        self.data.groupBy("cluster").count().orderBy("cluster").show()

    def eval_clustering(self) -> None:
        silhouette_score = ClusteringEvaluator(
            predictionCol="cluster", featuresCol="features"
        ).evaluate(self.data)
        print(f"Silhouette score: {silhouette_score}")

    def fit(self) -> None:
        pipeline = Pipeline(
            stages=[
                LogisticRegression(
                    featuresCol="features",
                    labelCol="cluster",
                    predictionCol="logreg_pred",
                    rawPredictionCol="raw",
                    probabilityCol="prob",
                ),
                LinearRegression(
                    featuresCol="features", labelCol="logreg_pred", predictionCol="pred"
                ),
            ]
        )
        print("Pipeline initialized successfully.")
        try:
            self.model = pipeline.fit(self.train)
            print("Pipeline fitted successfully.")
        except:
            print(traceback.format_exc())
            sys.exit(1)

    def predict(self, data: DataFrame) -> DataFrame:
        return self.model.transform(data)

    def print_metrics_train(self):
        print("Train metrics:")
        cls_summary = self.model.stages[0].summary
        reg_summary = self.model.stages[1].summary
        print(f"Classification accuracy: {cls_summary.accuracy:.2f}")
        print(f"Weighted precision: {cls_summary.weightedPrecision:.2f}")
        print(f"Weighted recall: {cls_summary.weightedRecall:.2f}")
        self.save_training_history(
            cls_summary.objectiveHistory,
            "Classification objective",
            os.path.join(self.plot_path, "cls_training.png"),
        )
        print(f"Regression MSE: {reg_summary.meanSquaredError:.2f}")
        print(f"Regression MAE: {reg_summary.meanAbsoluteError:.2f}")
        print(f"Explained variance: {reg_summary.explainedVariance:.2f}")
        self.save_training_history(
            cls_summary.objectiveHistory,
            "Regression objective",
            os.path.join(self.plot_path, "reg_training.png"),
        )

    def print_metrics_test(self):
        self.test = self.model.transform(self.test)
        self.test.select(
            ["product_id", "product_name", "prob", "logreg_pred", "cluster"]
        ).show(20, False)
        print("Test metrics:")
        mcl_eval = MulticlassClassificationEvaluator(
            predictionCol="logreg_pred", labelCol="cluster"
        )
        print(
            f"Classification accuracy: {mcl_eval.evaluate(self.test, {mcl_eval.metricName: 'accuracy'})}"
        )
        print(
            f"Weighted precision: {mcl_eval.evaluate(self.test, {mcl_eval.metricName: 'weightedPrecision'})}"
        )
        print(
            f"Weighted recall: {mcl_eval.evaluate(self.test, {mcl_eval.metricName: 'weightedRecall'})}"
        )
        reg_eval = RegressionEvaluator(predictionCol="pred", labelCol="logreg_pred")
        print(
            f"Regression MSE: {reg_eval.evaluate(self.test, {reg_eval.metricName: 'mse'})}"
        )
        print(
            f"Regression MAE: {reg_eval.evaluate(self.test, {reg_eval.metricName: 'mae'})}"
        )

    def evaluate(self):
        self.load_data()
        self.eda()
        self.split_data()
        self.eval_clustering()
        self.fit()
        self.print_metrics_train()
        self.print_metrics_test()
        # self.spark.stop()
