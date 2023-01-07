from pandas import DataFrame
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.regression import RegressionExperiment
from sklearn2pmml import make_pmml_pipeline
from sklearn2pmml.pycaret import _escape

import numpy
import os
import pycaret
import sys

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris", "Wheat"]

def make_classification(df, estimator, name, **setup_params):
	X, y = split_csv(df)

	categories = numpy.unique(y)

	exp = ClassificationExperiment()
	exp.setup(data = df, target = y.name, session_id = 13, **setup_params)

	if estimator == "dt" or estimator == "rf":
		model = exp.create_model(estimator, min_samples_leaf = 10)
	else:
		model = exp.create_model(estimator)

	final_model = exp.finalize_model(model)

	pipeline = make_pmml_pipeline(final_model, target_fields = [y.name], escape_func = _escape)
	store_pkl(pipeline, name)

	yt = exp.predict_model(final_model, data = X)["prediction_label"]
	yt.name = y.name
	yt_proba = DataFrame(final_model.predict_proba(X), columns = ["probability(" + str(category) + ")" for category in categories])
	store_csv(pandas.concat((yt, yt_proba), axis = 1), name)

def make_clustering(df, estimator, name, **setup_params):
	X, y = split_csv(df)

	exp = ClusteringExperiment()
	exp.setup(data = X, session_id = 13, **setup_params)

	model = exp.create_model(estimator)

	pipeline = make_pmml_pipeline(model, active_fields = X.columns.values, escape_func = _escape)
	store_pkl(pipeline, name)

	yt = exp.predict_model(model, data = X)["Cluster"]
	yt.name = "cluster"
	yt = yt.apply(lambda x: x.replace("Cluster ", ""))
	store_csv(yt, name)

def make_regression(df, estimator, name, **setup_params):
	X, y = split_csv(df)

	exp = RegressionExperiment()
	exp.setup(data = df, target = y.name, session_id = 13, **setup_params)

	if estimator == "dt" or estimator == "rf":
		model = exp.create_model(estimator, min_samples_leaf = 10)
	else:
		model = exp.create_model(estimator)

	final_model = exp.finalize_model(model)

	pipeline = make_pmml_pipeline(final_model, target_fields = [y.name], escape_func = _escape)
	store_pkl(pipeline, name)

	yt = exp.predict_model(final_model, data = X)["prediction_label"]
	yt.name = y.name
	store_csv(yt, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	make_classification(audit_df, "rf", "PyCaretAudit", rare_to_value = 0.03, rare_value = "Other")

	audit_df = load_audit("AuditNA")

	make_classification(audit_df, "lr", "PyCaretAuditNA")

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	make_classification(iris_df, "rf", "PyCaretIris")

if "Wheat" in datasets:
	wheat_df = load_wheat("Wheat")

	make_clustering(wheat_df, "kmeans", "PyCaretWheat")

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	# XXX
	auto_df["cylinders"] = auto_df["cylinders"].astype(str)
	auto_df["model_year"] = auto_df["model_year"].astype(str)
	auto_df["origin"] = auto_df["origin"].astype(str)

	make_regression(auto_df, "rf", "PyCaretAuto")

	auto_df = load_auto("AutoNA")

	make_regression(auto_df, "lr", "PyCaretAutoNA")