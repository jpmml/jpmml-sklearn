import os
import sys

from pandas import DataFrame, Int64Dtype, Series
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.internal.pipeline import Pipeline
from pycaret.regression import RegressionExperiment
from sklearn2pmml import make_pmml_pipeline
from sklearn2pmml.pycaret import _escape

import numpy

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris", "Wheat"]

def extract_preproc_pipeline(pipeline):
	preproc_steps = []
	for step in pipeline.steps:
		if step[0] == "remove_outliers":
			return Pipeline(steps = preproc_steps)
		preproc_steps.append(step)
	raise ValueError()

def predict_outlier_mask(pipeline, X):
	remove_outliers = pipeline["remove_outliers"]
	remove_outliers._train_only = False
	remove_outliers.transformer._train_only = False

	preproc_pipeline = extract_preproc_pipeline(pipeline)

	X_preproc = preproc_pipeline.transform(X)

	# XXX
	if remove_outliers.transformer._estimator is None:
		remove_outliers.transformer.transform(X_preproc)

	return remove_outliers.transformer._estimator.predict(X_preproc)

def make_classification(df, estimator, name, **setup_params):
	X, y = split_csv(df)

	categories = numpy.unique(y)

	exp = ClassificationExperiment()
	exp.setup(data = df, target = y.name, session_id = 13, **setup_params)

	if estimator == "dt" or estimator == "rf":
		model = exp.create_model(estimator, min_samples_leaf = 10)
	else:
		model = exp.create_model(estimator)

	calibrated_model = exp.calibrate_model(model)

	pipeline = exp.finalize_model(calibrated_model)

	if name.endswith("NA"):
		pipeline = make_pmml_pipeline(pipeline, target_fields = [y.name])

	store_pkl(pipeline, name, escape_func = _escape)

	yt = Series(pipeline.predict(X), name = y.name)
	yt_proba = DataFrame(pipeline.predict_proba(X), columns = ["probability(" + str(category) + ")" for category in categories])
	store_csv(pandas.concat((yt, yt_proba), axis = 1), name)

	if setup_params.get("remove_outliers", False):
		name = name.replace("PyCaret", "PyCaretMasked")

		mask = predict_outlier_mask(pipeline, X)

		yt[mask == -1] = None
		yt_proba.loc[mask == -1, :] = numpy.NaN

		store_pkl(pipeline, name, escape_func = None)

		store_csv(pandas.concat((yt, yt_proba), axis = 1), name)

def make_clustering(df, estimator, name, **setup_params):
	X, y = split_csv(df)

	exp = ClusteringExperiment()
	exp.setup(data = X, session_id = 13, **setup_params)

	model = exp.create_model(estimator)

	# XXX
	model.cluster_centers_ = model.cluster_centers_.astype(float)

	pipeline = make_pmml_pipeline(model)

	store_pkl(pipeline, name, escape_func = _escape)

	yt = Series(pipeline.predict(X), name = "cluster")
	store_csv(yt, name)

def make_regression(df, estimator, name, **setup_params):
	X, y = split_csv(df)

	exp = RegressionExperiment()
	exp.setup(data = df, target = y.name, session_id = 13, **setup_params)

	if estimator == "dt" or estimator == "rf":
		model = exp.create_model(estimator, min_samples_leaf = 10)
	else:
		model = exp.create_model(estimator)

	pipeline = exp.finalize_model(model)

	if name.endswith("NA"):
		pipeline = make_pmml_pipeline(pipeline, target_fields = [y.name])

	store_pkl(pipeline, name, escape_func = _escape)

	yt = Series(pipeline.predict(X), name = y.name)
	store_csv(yt, name)

	if setup_params.get("remove_outliers", False):
		name = name.replace("PyCaret", "PyCaretMasked")

		mask = predict_outlier_mask(pipeline, X)

		yt[mask == -1] = numpy.NaN

		store_pkl(pipeline, name, escape_func = None)

		store_csv(yt, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	make_classification(audit_df, "rf", "PyCaretAudit", rare_to_value = 0.03, rare_value = "Other")

	audit_df = load_audit("AuditNA")
	audit_df = audit_df.drop("Deductions", axis = 1)

	# XXX: Downgrade to fix_imbalance = False
	make_classification(audit_df, "lr", "PyCaretAuditNA", normalize = True, normalize_method = "robust", fix_imbalance = True)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	make_classification(iris_df, "lr", "PyCaretIris", remove_outliers = True, polynomial_features = True, polynomial_degree = 2, low_variance_threshold = 0.5)

if "Wheat" in datasets:
	wheat_df = load_wheat("Wheat")

	make_clustering(wheat_df, "kmeans", "PyCaretWheat")

if "Auto" in datasets:
	cat_cols = ["cylinders", "model_year", "origin"]
	
	auto_df = load_auto("Auto")

	for cat_col in cat_cols:
		auto_df[cat_col] = auto_df[cat_col].astype(str)

	make_regression(auto_df, "rf", "PyCaretAuto", remove_outliers = True, remove_multicollinearity = True, multicollinearity_threshold = 0.75)

	auto_df = load_auto("AutoNA")

	for cat_col in cat_cols:
		auto_df[cat_col] = auto_df[cat_col].astype(Int64Dtype())

	make_regression(auto_df, "lr", "PyCaretAutoNA", feature_selection = True, feature_selection_method = "classic", n_features_to_select = 0.85)
