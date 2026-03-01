import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from causalml.inference.meta import BaseSRegressor
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml.pipeline import PMMLPipeline

import joblib
import numpy
import pandas

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Email"]

def build_email(email_df, regressor, name):
	email_X, email_y = split_csv(email_df)

	cat_cols = ["channel", "zip_code"]
	binary_cols = ["mens", "newbie", "womens"]
	cont_cols = ["history", "recency"]

	treatment_transformer = ColumnTransformer([
		("keep", OrdinalEncoder(), ["segment"])
	], remainder = "drop")

	features_transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), cat_cols),
		("binary", "passthrough", binary_cols),
		("cont", "passthrough", cont_cols)
	], remainder = "drop")

	union = FeatureUnion([
		("treatment", treatment_transformer),
		("features", features_transformer)
	])
	union.set_output(transform = "pandas")

	email_Xt = union.fit_transform(email_X)

	email_treatment = email_X.iloc[:, 0]
	email_features = email_Xt.iloc[:, 1:]

	regressor.fit(email_features, email_treatment, email_y)

	pipeline = PMMLPipeline([
		("transformer", union),
		("regressor", regressor)
	])
	pipeline.active_fields = numpy.asarray(email_X.columns.values)
	pipeline.target_fields = numpy.asarray(["uplift"])
	store_pkl(pipeline, name)

	uplift = DataFrame(regressor.predict(email_features, email_treatment, email_y), columns = ["uplift"])
	store_csv(uplift, name)

if "Email" in datasets:
	email_df = load_csv("Email")

	email_binary_df = email_df.head(2000 + 2000)

	build_email(email_binary_df, BaseSRegressor(DecisionTreeRegressor(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSRegressorEmail")
	build_email(email_binary_df, BaseSRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingSRegressorEmail")
	build_email(email_binary_df, BaseSRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSRegressorEmail")
