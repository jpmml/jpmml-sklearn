import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from optbinning import BinningProcess, Scorecard
from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn2pmml.decoration import Alias
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import ExpressionTransformer

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris"]

def make_binning_process(cont_cols, cat_cols):
	return BinningProcess(variable_names = (cont_cols + cat_cols), categorical_variables = cat_cols)

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	cont_cols = ["Age", "Hours", "Income"]
	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]

	audit_X = audit_X[cont_cols + cat_cols]

	binning_process = make_binning_process(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("binning_process", binning_process),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

def build_ob_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	mapper = DataFrameMapper([
		(["Age"], [Alias(ExpressionTransformer("-999 if pandas.isnull(X[0]) else (-999 if (X[0] < 21 or X[0] > 65) else X[0])", dtype = int), name = "clean(Age)"), BinningProcess(variable_names = ["clean(Age)"], special_codes = [-999], binning_transform_params = {"clean(Age)" : {"metric" : "event_rate"}})]),
		(["Hours"], BinningProcess(variable_names = ["Hours"], binning_transform_params = {"Hours" : {"metric" : "event_rate"}})),
		(["Income"], BinningProcess(variable_names = ["Income"], binning_transform_params = {"Income" : {"metric" : "woe"}}))
	])

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

def build_scorecard_audit(audit_df, name, **scorecard_params):
	audit_X, audit_y = split_csv(audit_df)

	cont_cols = ["Age", "Hours", "Income"]
	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]

	audit_X = audit_X[cont_cols + cat_cols]

	binning_process = make_binning_process(cont_cols, cat_cols)
	estimator = LogisticRegression()

	scorecard = Scorecard(binning_process = binning_process, estimator = estimator, **scorecard_params)

	pipeline = PMMLPipeline([
		("scorecard", scorecard)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)

	if scorecard.scaling_method is not None:
		adjusted = DataFrame(scorecard.score(audit_X), columns = ["Adjusted"])
	else:
		adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	build_audit(audit_df, DecisionTreeClassifier(random_state = 13), "BinningProcessAudit")
	build_ob_audit(audit_df, DecisionTreeClassifier(random_state = 13), "OptimalBinningAudit")
	build_scorecard_audit(audit_df, "ScorecardAudit")
	build_scorecard_audit(audit_df, "ScaledScorecardAudit", scaling_method = "pdo_odds", scaling_method_params = {"pdo" : 20, "odds" : 50, "scorecard_points" : 600})

	audit_df = load_audit("AuditNA")

	build_audit(audit_df, LogisticRegression(), "BinningProcessAuditNA")
	build_ob_audit(audit_df, LogisticRegression(), "OptimalBinningAuditNA")

def build_iris(iris_df, classifier, name):
	iris_X, iris_y = split_csv(iris_df)

	cont_cols = list(iris_X.columns.values)
	cat_cols = []

	binning_process = make_binning_process(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("binning_process", binning_process),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_iris(iris_df, LogisticRegression(), "BinningProcessIris")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

	auto_X = auto_X[cont_cols + cat_cols]

	binning_process = make_binning_process(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("binning_process", binning_process),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

def build_scorecard_auto(auto_df, name, **scorecard_params):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

	auto_X = auto_X[cont_cols + cat_cols]

	binning_process = make_binning_process(cont_cols, cat_cols)
	estimator = HuberRegressor()

	scorecard = Scorecard(binning_process = binning_process, estimator = estimator, **scorecard_params)

	pipeline = PMMLPipeline([
		("scorecard", scorecard)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)

	if scorecard.scaling_method is not None:
		mpg = DataFrame(scorecard.score(auto_X), columns = ["mpg"])
	else:
		mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, DecisionTreeRegressor(random_state = 13), "BinningProcessAuto")
	build_scorecard_auto(auto_df, "ScorecardAuto")
	build_scorecard_auto(auto_df, "ScaledScorecardAuto", scaling_method = "min_max", scaling_method_params = {"min" : 0, "max" : 100}, intercept_based = True, reverse_scorecard = True)

	auto_df = load_auto("AutoNA")

	build_auto(auto_df, LinearRegression(), "BinningProcessAutoNA")
	build_scorecard_auto(auto_df, "ScorecardAutoNA")
	build_scorecard_auto(auto_df, "ScaledScorecardAutoNA", scaling_method = "min_max", scaling_method_params = {"min" : 0, "max" : 100})
