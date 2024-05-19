import os
import sys

from pandas import DataFrame
from interpret.glassbox import ClassificationTree, LinearRegression, ExplainableBoostingClassifier, ExplainableBoostingRegressor, LogisticRegression, RegressionTree
from sklearn2pmml.pipeline import PMMLPipeline

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Auto", "Iris", "Versicolor"]

def build_versicolor(versicolor_df, classifier, name):
	versicolor_X, versicolor_y = split_csv(versicolor_df)

	pipeline = PMMLPipeline([
		("classifier", classifier)
	])
	pipeline.fit(versicolor_X, versicolor_y)
	# XXX
	#pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Versicolor" in datasets:
	versicolor_df = load_versicolor("Versicolor")

	build_versicolor(versicolor_df, ExplainableBoostingClassifier(max_bins = 5, max_interaction_bins = 5, random_state = 13), "ExplainableBoostingClassifierVersicolor")

def build_iris(iris_df, classifier, name):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = PMMLPipeline([
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	# XXX
	#pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_iris(iris_df, ClassificationTree(max_depth = 3, random_state = 13), "ClassificationTreeIris")
	build_iris(iris_df, LogisticRegression(), "LogisticRegressionIris")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	pipeline = PMMLPipeline([
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	# XXX
	#pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	cat_cols = ["cylinders", "model_year", "origin"]
	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

	build_auto(auto_df, ExplainableBoostingRegressor(max_bins = 11, max_interaction_bins = 7, random_state = 13), "ExplainableBoostingRegressorAuto")

	auto_df[cat_cols] = auto_df[cat_cols].astype("category")

	build_auto(auto_df, LinearRegression(), "LinearRegressionAuto")
	build_auto(auto_df, RegressionTree(max_depth = 5, random_state = 13), "RegressionTreeAuto")
