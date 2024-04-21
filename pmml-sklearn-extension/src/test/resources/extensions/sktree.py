import os
import sys

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline
from sktree.tree import ObliqueDecisionTreeRegressor

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["cylinders", "model_year", "origin"]),
		("cont", "passthrough", ["acceleration", "displacement", "horsepower", "weight"])
	])

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

auto_df = load_auto("Auto")

build_auto(auto_df, ObliqueDecisionTreeRegressor(min_samples_leaf = 5, random_state = 13), "ObliqueDecisionTreeAuto")