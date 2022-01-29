from h2o import H2OFrame
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from sklearn.compose import ColumnTransformer
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.h2o import H2OFrameCreator
from sklearn_pandas import DataFrameMapper

import h2o
import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

h2o.init()
h2o.connect()

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["Age", "Hours", "Income"]] +
		[([column], CategoricalDomain()) for column in ["Employment", "Education", "Marital", "Occupation", "Gender", "Deductions"]]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("uploader", H2OFrameCreator()),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, H2OFrame(audit_y.to_frame(), column_types = ["categorical"]))
	# XXX
	#if isinstance(classifier, H2OXGBoostEstimator):
	#	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	#else:
	#	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	classifier = pipeline._final_estimator
	store_mojo(classifier, name)
	store_pkl(pipeline, name)
	adjusted = pipeline.predict(audit_X)
	adjusted.set_names(["h2o(Adjusted)", "probability(0)", "probability(1)"])
	store_csv(adjusted.as_data_frame(), name)

audit_df = load_audit("Audit")

audit_df["Adjusted"] = audit_df["Adjusted"].astype(str)

build_audit(audit_df, H2OGradientBoostingEstimator(distribution = "bernoulli", ntrees = 17), "H2OGradientBoostingAudit")
build_audit(audit_df, H2OGeneralizedLinearEstimator(family = "binomial"), "H2OLogisticRegressionAudit")
build_audit(audit_df, H2ORandomForestEstimator(distribution = "bernoulli", seed = 13), "H2ORandomForestAudit")
build_audit(audit_df, H2OXGBoostEstimator(ntrees = 17, seed = 13), "H2OXGBoostAudit")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	transformer = ColumnTransformer(
		[(column, CategoricalDomain(), [column]) for column in ["cylinders", "model_year", "origin"]] +
		[(column, ContinuousDomain(), [column]) for column in ["displacement", "horsepower", "weight", "acceleration"]]
	)
	pipeline = PMMLPipeline([
		("transformer", transformer),
		("uploader", H2OFrameCreator(column_names = ["cylinders", "model_year", "origin", "displacement", "horsepower", "weight", "acceleration"], column_types = ["enum", "enum", "enum", "numeric", "numeric", "numeric", "numeric"])),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, H2OFrame(auto_y.to_frame()))
	if isinstance(regressor, H2OXGBoostEstimator):
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	regressor = pipeline._final_estimator
	store_mojo(regressor, name)
	store_pkl(pipeline, name)
	mpg = pipeline.predict(auto_X)
	mpg.set_names(["mpg"])
	store_csv(mpg.as_data_frame(), name)

auto_df = load_auto("Auto")

auto_df["cylinders"] = auto_df["cylinders"].astype(int)
auto_df["model_year"] = auto_df["model_year"].astype(int)
auto_df["origin"] = auto_df["origin"].astype(int)

build_auto(auto_df, H2OGradientBoostingEstimator(distribution = "gaussian", ntrees = 17), "H2OGradientBoostingAuto")
build_auto(auto_df, H2OGeneralizedLinearEstimator(family = "gaussian"), "H2OLinearRegressionAuto")
build_auto(auto_df, H2ORandomForestEstimator(distribution = "gaussian", seed = 13), "H2ORandomForestAuto")
build_auto(auto_df, H2OXGBoostEstimator(ntrees = 17, seed = 13), "H2OXGBoostAuto")

h2o.shutdown()