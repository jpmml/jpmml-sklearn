import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from pandas import CategoricalDtype, DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost.sklearn import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

datasets = ["Audit", "Auto"]

def make_dataframe_mapper(cont_cols, cat_cols):
	mapper = DataFrameMapper(
		[([cont_col], [ContinuousDomain()]) for cont_col in cont_cols] +
		[([cat_col], [CategoricalDomain(dtype = CategoricalDtype())]) for cat_col in cat_cols]
	, input_df = True, df_out = True)

	return mapper

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	cont_cols = ["Age", "Hours", "Income"]
	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]

	mapper = make_dataframe_mapper(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit", stringify = True)

	build_audit(audit_df, XGBClassifier(n_estimators = 71, objective = "binary:logistic", enable_categorical = True, tree_method = "hist", random_state = 13), "XGBAuditCat")
	build_audit(audit_df, XGBRFClassifier(objective = "binary:logistic", max_depth = 5, enable_categorical = True, tree_method = "hist", random_state = 13), "XGBRFAuditCat")

	audit_na_df = load_audit("AuditNA")

	build_audit(audit_na_df, XGBClassifier(n_estimators = 71, objective = "binary:logistic", booster = "dart", enable_categorical = True, tree_method = "hist", random_state = 13), "XGBAuditCatNA")
	build_audit(audit_na_df, XGBRFClassifier(objective = "binary:logistic", booster = "dart", max_depth = 5, enable_categorical = True, tree_method = "hist", random_state = 13), "XGBRFAuditCatNA")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

	mapper = make_dataframe_mapper(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, XGBRegressor(objective = "reg:squarederror", enable_categorical = True, tree_method = "hist", random_state = 13), "XGBAutoCat")
	build_auto(auto_df, XGBRFRegressor(objective = "reg:squarederror", max_depth = 3, enable_categorical = True, tree_method = "hist", random_state = 13), "XGBRFAutoCat")

	auto_na_df = load_auto("AutoNA")

	build_auto(auto_na_df, XGBRegressor(objective = "reg:squarederror", booster = "dart", enable_categorical = True, tree_method = "hist", random_state = 13), "XGBAutoCatNA")
	build_auto(auto_na_df, XGBRFRegressor(objective = "reg:squarederror", booster = "dart", max_depth = 3, enable_categorical = True, tree_method = "hist", random_state = 13), "XGBRFAutoCatNA")

def build_multi_auto(auto_df, regressor, name):
	auto_X, auto_y = split_multi_csv(auto_df, ["acceleration", "mpg"])

	cont_cols = ["displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

	mapper = make_dataframe_mapper(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	acceleration_mpg = DataFrame(pipeline.predict(auto_X), columns = ["acceleration", "mpg"])
	store_csv(acceleration_mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_multi_auto(auto_df, XGBRegressor(objective = "reg:squarederror", enable_categorical = True, tree_method = "hist", random_state = 13), "MultiXGBAutoCat")
	build_multi_auto(auto_df, XGBRFRegressor(objective = "reg:squarederror", enable_categorical = True, tree_method = "hist", random_state = 13), "MultiXGBRFAutoCat")
