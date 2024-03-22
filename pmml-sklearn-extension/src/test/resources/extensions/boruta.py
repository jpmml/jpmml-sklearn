import os
import sys

from boruta import BorutaPy
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn2pmml.pipeline import PMMLPipeline

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

def build_auto(auto_df, name):
	auto_X, auto_y = split_csv(auto_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["cylinders", "model_year", "origin"]),
		("cont", StandardScaler(), ["acceleration", "displacement", "horsepower", "weight"])
	])

	selector = BorutaPy(RandomForestRegressor(max_depth = 3, random_state = 13), n_estimators = 11, perc = 80, verbose = 2)

	regressor = LinearRegression()

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("selector", selector),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

auto_df = load_auto("Auto")

build_auto(auto_df, "BorutaPyAuto")
