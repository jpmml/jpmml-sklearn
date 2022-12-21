from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.statsmodels import StatsModelsRegressor
from statsmodels.api import OLS, Poisson, WLS

import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

def make_mapper(cont_cols, cat_cols):
	return DataFrameMapper([
		(cont_cols, None),
		(cat_cols, OneHotEncoder())
	])

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

	mapper = make_mapper(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

auto_df = load_auto("Auto")

build_auto(auto_df, StatsModelsRegressor(OLS), "OLSAuto")
build_auto(auto_df, StatsModelsRegressor(WLS), "WLSAuto")

def build_visit(visit_df, regressor, name):
	visit_X, visit_y = split_csv(visit_df)

	cont_cols = ["age", "educ", "hhninc"]
	binary_cols = ["female", "kids", "married", "outwork", "self"]
	cat_cols = ["edlevel"]

	mapper = make_mapper(cont_cols, binary_cols + cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(visit_X, visit_y)
	store_pkl(pipeline, name)
	docvis = DataFrame(pipeline.predict(visit_X), columns = ["docvis"])
	store_csv(docvis, name)

visit_df = load_visit("Visit")

build_visit(visit_df, StatsModelsRegressor(Poisson), "PoissonVisit")