import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from ngboost import NGBRegressor
from ngboost.scores import LogScore, MLE
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import numpy

numpy.random.seed(13)

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Auto"]

def build_auto(auto_df, regressor, name, noisify_scales = False):
	auto_X, auto_y = split_csv(auto_df)

	cat_cols = ["cylinders", "model_year", "origin"]
	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

	transformer = ColumnTransformer(
		[(cat_col, make_pipeline(CategoricalDomain(), OneHotEncoder()), [cat_col]) for cat_col in cat_cols] +
		[(cont_col, ContinuousDomain(), [cont_col]) for cont_col in cont_cols]
	)

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	if noisify_scales:
		regressor.scalings = [scaling * (0.8 + 0.4 * numpy.random.random()) for scaling in regressor.scalings]
	# XXX
	regressor.fitted_ = True
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, NGBRegressor(n_estimators = 17, learning_rate = 0.1, Score = MLE, random_state = 13), "NGBoostAuto")
	build_auto(auto_df, NGBRegressor(n_estimators = 17, learning_rate = 0.5, Score = LogScore, random_state = 13), "NGBoostWeightedAuto", noisify_scales = True)
