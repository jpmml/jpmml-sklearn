from optbinning import BinningProcess, OptimalBinning
from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.optbinning import OptimalBinningWrapper

import os
import sys

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = ["Audit", "Auto", "Iris"]

def make_binning_process(cont_cols, cat_cols):
	return BinningProcess(variable_names = (cont_cols + cat_cols), categorical_variables = cat_cols)

def make_optimal_binning(metric):
	return OptimalBinningWrapper(OptimalBinning(dtype = "numerical"), metric = metric)

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	cont_cols = ["Age", "Hours", "Income"]
	#cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
	cat_cols = []

	audit_X = audit_X[cont_cols + cat_cols]

	processor = make_binning_process(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("processor", processor),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

def build_audit_ob(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	mapper = DataFrameMapper([
		(["Age"], make_optimal_binning(metric = "event_rate")),
		(["Hours"], make_optimal_binning(metric = "event_rate")),
		(["Income"], make_optimal_binning(metric = "woe"))
	])

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	build_audit(audit_df, DecisionTreeClassifier(random_state = 13), "BinningProcessAudit")
	build_audit_ob(audit_df, DecisionTreeClassifier(random_state = 13), "OptimalBinningAudit")

	audit_df = load_audit("AuditNA")

	build_audit(audit_df, LogisticRegression(), "BinningProcessAuditNA")
	build_audit_ob(audit_df, LogisticRegression(), "OptimalBinningAuditNA")

def build_iris(iris_df, classifier, name):
	iris_X, iris_y = split_csv(iris_df)

	cont_cols = list(iris_X.columns.values)
	cat_cols = []

	processor = make_binning_process(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("processor", processor),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_iris(iris_df, LogisticRegression(), "BinningProcessIris")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	#cat_cols = ["cylinders", "model_year", "origin"]
	cat_cols = []

	auto_X = auto_X[cont_cols + cat_cols]

	processor = make_binning_process(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("processor", processor),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, DecisionTreeRegressor(random_state = 13), "BinningProcessAuto")

	auto_df = load_auto("AutoNA")

	build_auto(auto_df, LinearRegression(), "BinningProcessAutoNA")
