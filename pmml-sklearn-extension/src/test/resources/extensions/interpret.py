import os
import sys

from pandas import DataFrame
from interpret.glassbox import LinearRegression, RegressionTree
from sklearn2pmml.pipeline import PMMLPipeline

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Auto"]

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

	auto_df[cat_cols] = auto_df[cat_cols].astype("category")

	build_auto(auto_df, LinearRegression(), "LinearRegressionAuto")
	build_auto(auto_df, RegressionTree(max_depth = 5, random_state = 13), "RegressionTreeAuto")
