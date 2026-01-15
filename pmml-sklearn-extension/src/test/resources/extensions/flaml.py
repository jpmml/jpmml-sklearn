import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from flaml import tune, AutoML
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml import make_pmml_pipeline

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto"]

def small_forest():
	return {
		"n_estimators": {
			"domain" : tune.randint(lower = 7, upper = 31),
			"init_value" : 11
		},
		"max_leaves": {
			"domain" : tune.lograndint(lower = 4, upper = 32),
			"low_cost_init_value" : 4
		}
	}

def make_custom_hp():
	return {
		"rf" : small_forest()
	}

def make_transformer(cat_cols, cont_cols, binarize = False):
	return ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False) if binarize else "passthrough", cat_cols),
		("cont", "passthrough", cont_cols)
	])

def build_audit(audit_df, classifier, name, binarize = False):
	audit_X, audit_y = split_csv(audit_df)

	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
	cont_cols = ["Age", "Hours", "Income"]

	transformer = make_transformer(cat_cols, cont_cols, binarize = binarize)

	audit_Xt = transformer.fit_transform(audit_X)

	automl = AutoML()
	automl.fit(audit_Xt, audit_y, task = "classification", estimator_list = [classifier], custom_hp = make_custom_hp(), time_budget = 10)

	pipeline = make_pmml_pipeline(make_pipeline(transformer, automl.model), target_fields = ["Adjusted"])
	store_pkl(pipeline, name)
	adjusted = DataFrame(automl.model.predict(audit_Xt), columns = ["Adjusted"])
	adjusted_proba = DataFrame(automl.model.predict_proba(audit_Xt), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	build_audit(audit_df, "lrl1", "LRL1ClassifierAudit", binarize = True)
	build_audit(audit_df, "lrl2", "LRL2ClassifierAudit", binarize = True)
	build_audit(audit_df, "rf", "RandomForestEstimatorAudit", binarize = True)
	build_audit(audit_df, "svc", "SVCEstimatorAudit", binarize = True)

def build_auto(auto_df, regressor, name, binarize = False):
	auto_X, auto_y = split_csv(auto_df)

	cat_cols = ["cylinders", "model_year", "origin"]
	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

	transformer = make_transformer(cat_cols, cont_cols, binarize = binarize)

	auto_Xt = transformer.fit_transform(auto_X)

	automl = AutoML()
	automl.fit(auto_Xt, auto_y, task = "regression", estimator_list = [regressor], custom_hp = make_custom_hp(), time_budget = 10)

	pipeline = make_pmml_pipeline(make_pipeline(transformer, automl.model), target_fields = ["mpg"])
	store_pkl(pipeline, name)
	mpg = DataFrame(automl.model.predict(auto_Xt), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, "enet", "ElasticNetEstimatorAuto", binarize = True)
	build_auto(auto_df, "lassolars", "LassoLarsEstimatorAuto", binarize = True)
	build_auto(auto_df, "rf", "RandomForestEstimatorAuto", binarize = True)
