from common import *

from h2o import H2OFrame
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.compose import ColumnTransformer
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.h2o import H2OFrameCreator
from sklearn_pandas import DataFrameMapper

import h2o

h2o.init()
h2o.connect()

audit_X, audit_y = load_audit("Audit")

def build_audit(classifier, name):
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
	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	classifier = pipeline._final_estimator
	store_mojo(classifier, name)
	store_pkl(pipeline, name)
	adjusted = pipeline.predict(audit_X)
	adjusted.set_names(["h2o(Adjusted)", "probability(0)", "probability(1)"])
	store_csv(adjusted.as_data_frame(), name)

build_audit(H2OGradientBoostingEstimator(distribution = "bernoulli", ntrees = 17), "H2OGradientBoostingAudit")
build_audit(H2OGeneralizedLinearEstimator(family = "binomial"), "H2OLogisticRegressionAudit")
build_audit(H2ORandomForestEstimator(distribution = "bernoulli", seed = 13), "H2ORandomForestAudit")

auto_X, auto_y = load_auto("Auto")

auto_X["cylinders"] = auto_X["cylinders"].astype(int)
auto_X["model_year"] = auto_X["model_year"].astype(int)
auto_X["origin"] = auto_X["origin"].astype(int)

def build_auto(regressor, name):
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
	pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	regressor = pipeline._final_estimator
	store_mojo(regressor, name)
	store_pkl(pipeline, name)
	mpg = pipeline.predict(auto_X)
	mpg.set_names(["mpg"])
	store_csv(mpg.as_data_frame(), name)

build_auto(H2OGradientBoostingEstimator(distribution = "gaussian", ntrees = 17), "H2OGradientBoostingAuto")
build_auto(H2OGeneralizedLinearEstimator(family = "gaussian"), "H2OLinearRegressionAuto")
build_auto(H2ORandomForestEstimator(distribution = "gaussian", seed = 13), "H2ORandomForestAuto")

h2o.shutdown()