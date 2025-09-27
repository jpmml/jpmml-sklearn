import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from h2o import H2OFrame
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.h2o import H2OFrameConstructor
from sklearn_pandas import DataFrameMapper

import h2o

h2o.init()
h2o.connect()

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto"]

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	mapper = DataFrameMapper([
		(["Age", "Hours"], ContinuousDomain()),
		(["Income"], ContinuousDomain()),
		(["Employment", "Education", "Marital", "Occupation", "Gender", "Deductions"], CategoricalDomain())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("uploader", H2OFrameConstructor()),
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
	embed_stored_mojo(classifier, 50000)
	store_pkl(pipeline, name, flavour = "dill")
	adjusted = pipeline.predict(audit_X)
	adjusted.set_names(["h2o(Adjusted)", "probability(0)", "probability(1)"])
	store_csv(adjusted.as_data_frame(), name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	audit_df["Adjusted"] = audit_df["Adjusted"].astype(str)

	build_audit(audit_df, H2OGradientBoostingEstimator(distribution = "bernoulli", ntrees = 17), "H2OGradientBoostingAudit")
	build_audit(audit_df, H2OGeneralizedLinearEstimator(family = "binomial"), "H2OLogisticRegressionAudit")
	build_audit(audit_df, H2ORandomForestEstimator(distribution = "bernoulli", seed = 13), "H2ORandomForestAudit")
	build_audit(audit_df, H2OXGBoostEstimator(ntrees = 17, seed = 13), "H2OXGBoostAudit")

def build_auto(auto_df, regressor, name, classes = None):
	auto_X, auto_y = split_csv(auto_df)

	cat_cols = ["cylinders", "model_year", "origin"]
	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

	transformer = ColumnTransformer([
		("cat", CategoricalDomain(), cat_cols),
		("cont", ContinuousDomain(), cont_cols)
	])
	pipeline = PMMLPipeline([
		("transformer", transformer),
		("uploader", H2OFrameConstructor(column_names = cat_cols + cont_cols, column_types = ["enum"] * len(cat_cols) + ["numeric"] * len(cont_cols))),
		("regressor", regressor)
	])
	if classes:
		pipeline.fit(auto_X, H2OFrame(auto_y.to_frame(), column_types = ["categorical"]))
	else:
		pipeline.fit(auto_X, H2OFrame(auto_y.to_frame()))
	if isinstance(regressor, H2OXGBoostEstimator):
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		if classes:
			pass
		else:
			pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	regressor = pipeline._final_estimator
	if classes:
		regressor.pmml_classes_ = classes
	store_mojo(regressor, name)
	embed_stored_mojo(regressor, 50000)
	store_pkl(pipeline, name, flavour = "dill")
	mpg = pipeline.predict(auto_X)
	if classes:
		mpg.set_names(["bin(mpg)"] + ["probability({})".format(clazz) for clazz in classes])
	else:
		mpg.set_names(["mpg"])
	mpg = mpg.as_data_frame()
	if classes:
		class_mapping = {idx : clazz for idx, clazz in enumerate(classes)}
		mpg["bin(mpg)"] = numpy.vectorize(lambda x: class_mapping[x])(mpg["bin(mpg)"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	auto_df["cylinders"] = auto_df["cylinders"].astype(int)
	auto_df["model_year"] = auto_df["model_year"].astype(int)
	auto_df["origin"] = auto_df["origin"].astype(int)

	build_auto(auto_df, H2OGradientBoostingEstimator(distribution = "gaussian", ntrees = 17), "H2OGradientBoostingAuto")
	build_auto(auto_df, H2OGeneralizedLinearEstimator(family = "gaussian"), "H2OLinearRegressionAuto")
	build_auto(auto_df, H2ORandomForestEstimator(distribution = "gaussian", seed = 13), "H2ORandomForestAuto")
	build_auto(auto_df, H2OXGBoostEstimator(ntrees = 17, seed = 13), "H2OXGBoostAuto")

	auto_df.rename(columns = {"mpg" : "bin(mpg)"}, inplace = True)

	categories = ["bad", "poor", "fair", "good", "excellent"]

	binner = KBinsDiscretizer(n_bins = len(categories), encode = "ordinal", strategy = "kmeans")
	auto_df["bin(mpg)"] = binner.fit_transform(auto_df["bin(mpg)"].values.reshape(-1, 1)).astype(int)

	build_auto(auto_df, H2OGeneralizedLinearEstimator(family = "ordinal", seed = 13), "H2OOrdinalRegressionAuto", classes = categories)

h2o.shutdown()
