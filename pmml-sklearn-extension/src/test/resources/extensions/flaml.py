import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from flaml import tune, AutoML
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn2pmml import make_pmml_pipeline
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto"]

def make_fit_kwargs_by_estimator(cat_cols, cont_cols):
	return {
		"lgbm" : {
			"categorical_feature" : list(range(len(cat_cols)))
		},
		"xgboost" : {}
	}

def small_forest():
	return {
		"n_estimators" : {
			"domain" : tune.randint(lower = 7, upper = 31),
			"init_value" : 11
		},
		"max_leaves" : {
			"domain" : tune.lograndint(lower = 4, upper = 32),
			"low_cost_init_value" : 4
		}
	}

def make_custom_hp(cat_cols, cont_cols):
	xgb_feature_types = ["c"] * len(cat_cols) + ["q"] * len(cont_cols)

	xgb_config = {
		**small_forest(),
		"enable_categorical" : {
			"domain" : True,
			"init_value" : True
		},
		"feature_types" : {
			"domain" : xgb_feature_types,
			"init_value" : xgb_feature_types
		}
	}

	return {
		"extra_tree" : small_forest(),
		"histgb" : {
			**small_forest(),
			"categorical_features" : {
				"domain" : list(range(len(cat_cols))),
				"init_value" : list(range(len(cat_cols)))
			}
		},
		"lgbm" : small_forest(),
		"rf" : small_forest(),
		"xgboost" : xgb_config,
		"xgb_limitdepth" : xgb_config
	}

def make_transformer(cat_cols, cont_cols, binarize, standardize):
	transformer = ColumnTransformer(
		[(cat_col, make_pipeline(CategoricalDomain(), OneHotEncoder(sparse_output = False)) if binarize else CategoricalDomain(), [cat_col]) for cat_col in cat_cols] +
		[(cont_col, make_pipeline(ContinuousDomain(), StandardScaler()) if standardize else ContinuousDomain(), [cont_col]) for cont_col in cont_cols]
	)
	transformer.set_output(transform = "pandas")
	return transformer

def build_audit(audit_df, classifier, name, binarize = True, standardize = True):
	audit_X, audit_y = split_csv(audit_df)

	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
	cont_cols = ["Age", "Hours", "Income"]

	for cat_col in cat_cols:
		audit_X[cat_col] = audit_X[cat_col].astype(str) if binarize else audit_X[cat_col].astype("category")
	for cont_col in cont_cols:
		audit_X[cont_col] = audit_X[cont_col].astype(float)

	transformer = make_transformer(cat_cols, cont_cols, binarize = binarize, standardize = standardize)

	audit_Xt = transformer.fit_transform(audit_X)

	automl = AutoML()
	automl.fit(audit_Xt, audit_y, task = "classification", estimator_list = [classifier], fit_kwargs_by_estimator = make_fit_kwargs_by_estimator(cat_cols, cont_cols), custom_hp = make_custom_hp(cat_cols, cont_cols), time_budget = 10)

	pipeline = make_pmml_pipeline(make_pipeline(transformer, automl.model), target_fields = ["Adjusted"])
	store_pkl(pipeline, name)
	adjusted = DataFrame(automl.model.predict(audit_Xt), columns = ["Adjusted"])
	adjusted_proba = DataFrame(automl.model.predict_proba(audit_Xt), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	build_audit(audit_df, "extra_tree", "ExtraTreesEstimatorAudit", standardize = False)
	build_audit(audit_df, "histgb", "HistGradientBoostingEstimatorAudit", binarize = False, standardize = False)
	build_audit(audit_df, "lrl1", "LRL1ClassifierAudit")
	build_audit(audit_df, "lrl2", "LRL2ClassifierAudit")
	build_audit(audit_df, "rf", "RandomForestEstimatorAudit", standardize = False)
	build_audit(audit_df, "svc", "SVCEstimatorAudit")

def build_auto(auto_df, regressor, name, binarize = True, standardize = True):
	auto_X, auto_y = split_csv(auto_df)

	cat_cols = ["cylinders", "model_year", "origin"]
	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

	for cat_col in cat_cols:
		auto_X[cat_col] = auto_X[cat_col].astype(int) if binarize else auto_X[cat_col].astype("category")
	for cont_col in cont_cols:
		auto_X[cont_col] = auto_X[cont_col].astype(float)

	transformer = make_transformer(cat_cols, cont_cols, binarize = binarize, standardize = standardize)

	auto_Xt = transformer.fit_transform(auto_X)

	automl = AutoML()
	automl.fit(auto_Xt, auto_y, task = "regression", estimator_list = [regressor], fit_kwargs_by_estimator = make_fit_kwargs_by_estimator(cat_cols, cont_cols), custom_hp = make_custom_hp(cat_cols, cont_cols), time_budget = 10)

	pipeline = make_pmml_pipeline(make_pipeline(transformer, automl.model), target_fields = ["mpg"])
	store_pkl(pipeline, name)
	mpg = DataFrame(automl.model.predict(auto_Xt), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, "extra_tree", "ExtraTreesEstimatorAuto")
	build_auto(auto_df, "enet", "ElasticNetEstimatorAuto")
	build_auto(auto_df, "histgb", "HistGradientBoostingEstimatorAuto", binarize = False, standardize = False)
	build_auto(auto_df, "lassolars", "LassoLarsEstimatorAuto")
	build_auto(auto_df, "rf", "RandomForestEstimatorAuto", standardize = False)
