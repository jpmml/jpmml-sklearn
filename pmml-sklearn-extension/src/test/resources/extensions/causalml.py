import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from causalml.inference.meta import BaseSClassifier, BaseSRegressor
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

def build_email(email_df, estimator, name):
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

	estimator.fit(email_features, email_treatment, email_y)

	pipeline = PMMLPipeline([
		("transformer", union),
		("estimator", estimator)
	])
	pipeline.active_fields = numpy.asarray(email_X.columns.values)
	if name.endswith("Email"):
		pipeline.target_fields = numpy.asarray(["uplift(mens)", "uplift(womens)"])
	elif name.endswith("EmailBin"):
		pipeline.target_fields = numpy.asarray(["uplift"])
	else:
		raise ValueError()
	store_pkl(pipeline, name)

	uplift = DataFrame(estimator.predict(email_features, email_treatment, email_y), columns = pipeline.target_fields)
	store_csv(uplift, name)

if "Email" in datasets:
	email_df = load_csv("Email")

	build_email(email_df, BaseSRegressor(DecisionTreeRegressor(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSRegressorEmail")
	build_email(email_df, BaseSRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingSRegressorEmail")
	build_email(email_df, BaseSRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSRegressorEmail")

	email_binary_df = email_df.copy()
	email_binary_df["segment"] = email_binary_df["segment"].replace({
		"mens_email" : "email",
		"womens_email" : "email"
	})

	build_email(email_binary_df, BaseSClassifier(DecisionTreeClassifier(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSClassifierEmailBin")
	build_email(email_binary_df, BaseSClassifier(RandomForestClassifier(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSClassifierEmailBin")

	build_email(email_binary_df, BaseSRegressor(DecisionTreeRegressor(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSRegressorEmailBin")
	build_email(email_binary_df, BaseSRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingSRegressorEmailBin")
	build_email(email_binary_df, BaseSRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSRegressorEmailBin")
