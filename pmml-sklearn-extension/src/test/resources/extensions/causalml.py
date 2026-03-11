import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from causalml.inference.meta import BaseSClassifier, BaseSRegressor, BaseTClassifier, BaseTRegressor, BaseXClassifier, BaseXRegressor
from causalml.propensity import ElasticNetPropensityModel, LogisticRegressionPropensityModel
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
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

def to_binary(email_df):
	email_binary_df = email_df.copy()
	email_binary_df["segment"] = email_binary_df["segment"].replace({
		"mens_email" : "email",
		"womens_email" : "email"
	})
	return email_binary_df

def make_email_transformer():
	cat_cols = ["channel", "zip_code"]
	binary_cols = ["mens", "newbie", "womens"]
	cont_cols = ["history", "recency"]

	return ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), cat_cols),
		("binary", "passthrough", binary_cols),
		("cont", "passthrough", cont_cols)
	], remainder = "drop")

def build_email_single(email_df, estimator, name):
	email_X, email_y = split_csv(email_df)

	treatment_transformer = ColumnTransformer([
		("keep", OrdinalEncoder(), ["segment"])
	], remainder = "drop")

	features_transformer = make_email_transformer()

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

	build_email_single(email_df, BaseSClassifier(DecisionTreeClassifier(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSClassifierEmail")
	build_email_single(email_df, BaseSClassifier(RandomForestClassifier(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSClassifierEmail")

	build_email_single(email_df, BaseSRegressor(DecisionTreeRegressor(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSRegressorEmail")
	build_email_single(email_df, BaseSRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingSRegressorEmail")
	build_email_single(email_df, BaseSRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSRegressorEmail")

	email_binary_df = to_binary(email_df)

	build_email_single(email_binary_df, BaseSClassifier(DecisionTreeClassifier(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSClassifierEmailBin")
	build_email_single(email_binary_df, BaseSClassifier(RandomForestClassifier(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSClassifierEmailBin")

	build_email_single(email_binary_df, BaseSRegressor(DecisionTreeRegressor(max_depth = 9, random_state = 42), control_name = "control"), "DecisionTreeSRegressorEmailBin")
	build_email_single(email_binary_df, BaseSRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingSRegressorEmailBin")
	build_email_single(email_binary_df, BaseSRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestSRegressorEmailBin")

def build_email_propensity(email_df, estimator, name):
	email_X, email_y = split_csv(email_df)

	transformer = make_email_transformer()

	email_Xt = transformer.fit_transform(email_X)

	email_propensity = Series((email_X.iloc[:, 0] != "control").astype(int), name = "propensity")

	estimator.fit(email_Xt, email_propensity)

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("estimator", estimator)
	])
	pipeline.active_fields = numpy.asarray(email_X.columns.values)
	pipeline.target_fields = numpy.asarray(["propensity"])
	store_pkl(pipeline, name)

	propensity = DataFrame(estimator.predict(email_Xt), columns = pipeline.target_fields)
	store_csv(propensity, name)

if "Email" in datasets:
	email_df = load_csv("Email")

	build_email_propensity(email_df, ElasticNetPropensityModel(calibrate = True), "ElasticNetPropensityModelIsotonicEmail")
	build_email_propensity(email_df, ElasticNetPropensityModel(calibrate = False, clip_bounds = (0.05, 0.95)), "ElasticNetPropensityModelEmail")

	build_email_propensity(email_df, LogisticRegressionPropensityModel(calibrate = True), "LogisticRegressionPropensityModelIsotonicEmail")
	build_email_propensity(email_df, LogisticRegressionPropensityModel(calibrate = False, clip_bounds = (0.05, 0.95)), "LogisticRegressionPropensityModelEmail")

def build_email_parallel(email_df, estimator, name):
	email_X, email_y = split_csv(email_df)

	transformer = make_email_transformer()

	email_Xt = transformer.fit_transform(email_X)

	email_treatment = email_X.iloc[:, 0]

	if isinstance(estimator, (BaseXClassifier, BaseXRegressor)):
		estimator.model_p = LogisticRegression()

	estimator.fit(email_Xt, email_treatment, email_y)

	pipeline = PMMLPipeline([
		("transformer", transformer),
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

	if isinstance(estimator, (BaseXClassifier, BaseXRegressor)):
		propensity = dict()
		for group in estimator.t_groups:
			propensity[group] = estimator.propensity_model[group].predict_proba(email_Xt)[:, 1]
		uplift = DataFrame(estimator.predict(email_Xt, p = propensity), columns = pipeline.target_fields)
	else:
		uplift = DataFrame(estimator.predict(email_Xt), columns = pipeline.target_fields)
	store_csv(uplift, name)

if "Email" in datasets:
	email_df = load_csv("Email")

	build_email_parallel(email_df, BaseTClassifier(DecisionTreeClassifier(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeTClassifierEmail")
	build_email_parallel(email_df, BaseTClassifier(RandomForestClassifier(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestTClassifierEmail")

	build_email_parallel(email_df, BaseTRegressor(DecisionTreeRegressor(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeTRegressorEmail")
	build_email_parallel(email_df, BaseTRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingTRegressorEmail")
	build_email_parallel(email_df, BaseTRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestTRegressorEmail")

	email_binary_df = to_binary(email_df)

	build_email_parallel(email_binary_df, BaseTClassifier(DecisionTreeClassifier(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeTClassifierEmailBin")
	build_email_parallel(email_binary_df, BaseTClassifier(RandomForestClassifier(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestTClassifierEmailBin")

	build_email_parallel(email_binary_df, BaseTRegressor(DecisionTreeRegressor(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeTRegressorEmailBin")
	build_email_parallel(email_binary_df, BaseTRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingTRegressorEmailBin")
	build_email_parallel(email_binary_df, BaseTRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestTRegressorEmailBin")

if "Email" in datasets:
	email_df = load_csv("Email")

	build_email_parallel(email_df, BaseXClassifier(DecisionTreeClassifier(max_depth = 7, random_state = 42), DecisionTreeRegressor(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeXClassifierEmail")
	build_email_parallel(email_df, BaseXClassifier(RandomForestClassifier(n_estimators = 17, max_depth = 5, random_state = 42), RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestXClassifierEmail")

	build_email_parallel(email_df, BaseXRegressor(DecisionTreeRegressor(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeXRegressorEmail")
	build_email_parallel(email_df, BaseXRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingXRegressorEmail")
	build_email_parallel(email_df, BaseXRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestXRegressorEmail")

	email_binary_df = to_binary(email_df)

	build_email_parallel(email_binary_df, BaseXClassifier(DecisionTreeClassifier(max_depth = 7, random_state = 42), DecisionTreeRegressor(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeXClassifierEmailBin")
	build_email_parallel(email_binary_df, BaseXClassifier(RandomForestClassifier(n_estimators = 17, max_depth = 5, random_state = 42), RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestXClassifierEmailBin")

	build_email_parallel(email_binary_df, BaseXRegressor(DecisionTreeRegressor(max_depth = 7, random_state = 42), control_name = "control"), "DecisionTreeXRegressorEmailBin")
	build_email_parallel(email_binary_df, BaseXRegressor(GradientBoostingRegressor(n_estimators = 31, max_depth = 3, random_state = 42), control_name = "control"), "GradientBoostingXRegressorEmailBin")
	build_email_parallel(email_binary_df, BaseXRegressor(RandomForestRegressor(n_estimators = 17, max_depth = 5, random_state = 42), control_name = "control"), "RandomForestXRegressorEmailBin")
