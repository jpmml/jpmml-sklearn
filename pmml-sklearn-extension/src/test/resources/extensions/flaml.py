import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from flaml import AutoML
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml import make_pmml_pipeline

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Auto"]

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["cylinders", "model_year", "origin"]),
		("cont", "passthrough", ["acceleration", "displacement", "horsepower", "weight"])
	])

	auto_Xt = transformer.fit_transform(auto_X)

	automl = AutoML()
	automl.fit(auto_Xt, auto_y, task = "regression", estimator_list = [regressor], time_budget = 10)

	pipeline = make_pmml_pipeline(make_pipeline(transformer, automl.model), target_fields = ["mpg"])
	store_pkl(pipeline, name)
	mpg = DataFrame(automl.model.predict(auto_Xt), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, "enet", "ElasticNetEstimatorAuto")