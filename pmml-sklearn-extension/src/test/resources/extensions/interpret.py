import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from interpret.glassbox import ClassificationTree, LinearRegression, ExplainableBoostingClassifier, ExplainableBoostingRegressor, LogisticRegression, RegressionTree
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn2pmml.cross_reference import Recaller
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.postprocessing import FeatureExporter

import numpy

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris", "Versicolor"]

def build_audit(audit_df, classifier, name, predict_transformer = None):
	audit_X, audit_y = split_csv(audit_df)

	pipeline = PMMLPipeline([
		("classifier", classifier)
	], predict_transformer = predict_transformer)
	pipeline.fit(audit_X, audit_y)
	# XXX
	#pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
	cont_cols = ["Age", "Hours", "Income"]

	audit_df = audit_df[cat_cols + cont_cols + ["Adjusted"]]

	build_audit(audit_df, ExplainableBoostingClassifier(interactions = "3x", max_bins = 11, max_interaction_bins = 7, random_state = 13), "ExplainableBoostingClassifierAudit", predict_transformer = make_pipeline(Recaller(None, names = ["lookup(prepare(Education))", "lookup(bin(Age, 0))"]), FeatureExporter(names = ["ebm(Education)", "ebm(Age)"])))

if "Audit" in datasets:
	audit_df = load_audit("AuditNA")

	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
	cont_cols = ["Age", "Hours", "Income"]

	audit_df = audit_df[cat_cols + cont_cols + ["Adjusted"]]

	for cat_col in cat_cols:
		audit_df[cat_col] = audit_df[cat_col].replace({numpy.nan: None}).astype("category")

	build_audit(audit_df, ExplainableBoostingClassifier(interactions = "3x", max_bins = 5, max_interaction_bins = 7, random_state = 13), "ExplainableBoostingClassifierAuditNA")

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

	build_versicolor(versicolor_df, ExplainableBoostingClassifier(interactions = "2x", max_bins = 5, max_interaction_bins = 5, random_state = 13), "ExplainableBoostingClassifierVersicolor")

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

def build_auto(auto_df, regressor, name, predict_transformer = None):
	auto_X, auto_y = split_csv(auto_df)

	pipeline = PMMLPipeline([
		("regressor", regressor)
	], predict_transformer = predict_transformer)
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

	auto_df[cat_cols] = auto_df[cat_cols].astype("category")

	build_auto(auto_df, ExplainableBoostingRegressor(objective = "rmse_log", interactions = "5x", max_bins = 11, max_interaction_bins = 7, random_state = 13), "ExplainableBoostingRegressorAuto", predict_transformer = make_pipeline(Recaller(None, names = ["lookup(prepare(cylinders))", "lookup(bin(acceleration, 0))"]), FeatureExporter(names = ["ebm(cylinders)", "ebm(acceleration)"])))
	build_auto(auto_df, LinearRegression(), "LinearRegressionAuto")
	build_auto(auto_df, RegressionTree(max_depth = 5, random_state = 13), "RegressionTreeAuto")

if "Auto" in datasets:
	auto_df = load_auto("AutoNA")

	cat_cols = ["cylinders", "model_year", "origin"]
	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

	for cat_col in cat_cols:
		auto_df[cat_col] = auto_df[cat_col].astype(pandas.Int64Dtype()).astype("category")

	build_auto(auto_df, ExplainableBoostingRegressor(objective = "rmse", interactions = "5x", max_bins = 5, max_interaction_bins = 5, random_state = 13), "ExplainableBoostingRegressorAutoNA")
