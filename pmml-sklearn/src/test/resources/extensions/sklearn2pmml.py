import sys

from pandas import DataFrame, Series
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.cross_reference import make_memorizer_union, make_recaller_union
from sklearn2pmml.cross_reference import Memory, Memorizer, Recaller
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain, MultiAlias
from sklearn2pmml.ensemble import EstimatorChain, GBDTLMRegressor, GBDTLRClassifier, Link, SelectFirstClassifier, SelectFirstRegressor
from sklearn2pmml.expression import ExpressionClassifier, ExpressionRegressor
from sklearn2pmml.neural_network import MLPTransformer
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.postprocessing import BusinessDecisionTransformer
from sklearn2pmml.preprocessing import CutTransformer, LagTransformer, RollingAggregateTransformer, SelectFirstTransformer
from sklearn2pmml.ruleset import RuleSetClassifier
from sklearn2pmml.tree.chaid import CHAIDClassifier, CHAIDRegressor
from sklearn2pmml.util import Expression, Predicate

import numpy

from common import *

sys.path.append("../")

from main import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Airline", "Audit", "Auto", "Housing", "Iris", "Versicolor", "Wine"]

def make_bins(X, cols):
	result = dict()
	for col in cols:
		bins = numpy.nanquantile(X[col], q = [0.0, 0.25, 0.50, 0.75, 1.0], interpolation = "nearest")
		# Deduplicate
		bins = list(dict.fromkeys(bins))
		# Unbox Numpy float scalars to Python floats
		bins = [float(bin) for bin in bins]
		result[col] = bins
	return result

def make_bin_labels(bins):
	result = dict()
	for key, value in bins.items():
		result[key] = list(range(0, len(value) - 1))
	return result

def make_chaid_dataframe_mapper(cont_cols, cat_cols, bins = {}, labels = {}):
	return DataFrameMapper(
		[([cont_col], [ContinuousDomain(), CutTransformer(bins = bins[cont_col], labels = labels[cont_col])]) for cont_col in cont_cols] +
		[([cat_col], [CategoricalDomain()]) for cat_col in cat_cols]
	, df_out = True)

def build_chaid_audit(audit_df, name):
	audit_X, audit_y = split_csv(audit_df)

	cont_cols = ["Age", "Hours", "Income"]
	cat_cols = ["Education", "Employment", "Marital", "Occupation", "Gender"]

	bins = make_bins(audit_X, cont_cols)
	labels = make_bin_labels(bins)

	pipeline = PMMLPipeline([
		("mapper", make_chaid_dataframe_mapper(cont_cols, cat_cols, bins, labels)),
		("classifier", CHAIDClassifier(config = {"max_depth" : 9, "min_child_node_size" : 10}))
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	node_id = Series(pipeline._final_estimator.apply(audit_X), dtype = int, name = "nodeId")
	store_csv(node_id, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit", stringify = False)

	build_audit(audit_df, GBDTLRClassifier(RandomForestClassifier(n_estimators = 17, random_state = 13), LogisticRegression()), "GBDTLRAudit")

	audit_df = load_audit("Audit")

	build_multi_audit(audit_df, EstimatorChain([("gender", Link(DecisionTreeClassifier(max_depth = 5, random_state = 13), augment_funcs = ["predict_proba", "apply"]), str(True)), ("adjusted", LogisticRegression(), str(True))]), "MultiEstimatorChainAudit")

	build_chaid_audit(audit_df, "CHAIDAudit")

	audit_df = load_audit("AuditNA")

	build_chaid_audit(audit_df, "CHAIDAuditNA")

def _versicolor_func(petal_length, petal_width):
	if petal_length > 2.45:
		if petal_width <= 1.75:
			return 2.2824 # logit(49. / 54.)
		else:
			return -3.8067 # logit(1. / 46.)
	else:
		return -9.2102 # logit(0.0001)

def build_expr_versicolor(versicolor_df, name):
	versicolor_X, versicolor_y = split_csv(versicolor_df)

	class_exprs = {
		1: Expression("_versicolor_func(X['Petal.Length'], X['Petal.Width'])", function_defs = [_versicolor_func])
	}

	pipeline = PMMLPipeline([
		("classifier", ExpressionClassifier(class_exprs, normalization_method = "logit"))
	])
	pipeline.fit(versicolor_X, versicolor_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Versicolor" in datasets:
	versicolor_df = load_versicolor("Versicolor")

	build_versicolor(versicolor_df, GBDTLRClassifier(GradientBoostingClassifier(n_estimators = 11, random_state = 13), LogisticRegression(solver = "liblinear")), "GBDTLRVersicolor")

	build_expr_versicolor(versicolor_df, "ExpressionVersicolor")

def build_chaid_iris(iris_df, name):
	iris_X, iris_y = split_csv(iris_df)

	mapper = DataFrameMapper([
		(iris_X.columns.values, KBinsDiscretizer(n_bins = 5, encode = "ordinal", strategy = "kmeans"))
	])

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", CHAIDClassifier(config = {"max_depth" : 3, "min_child_node_size" : 10}))
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	node_id = Series(pipeline._final_estimator.apply(iris_X), dtype = int, name = "nodeId")
	store_csv(node_id, name)

def _iris_setosa_func(petal_length):
	if petal_length <= 2.45:
		return 1
	else:
		return 0

def _iris_versicolor_func(petal_length, petal_width):
	if petal_length > 2.45:
		if petal_width <= 1.75:
			return 0.90741
		else:
			return 0.02174
	else:
		return 0

def _iris_virginica_func(petal_length, petal_width):
	if petal_length > 2.45:
		if petal_width <= 1.75:
			return 0.09259
		else:
			return 0.97826
	else:
		return 0

def build_expr_iris(iris_df, name):
	iris_X, iris_y = split_csv(iris_df)

	class_exprs = {
		"setosa" : Expression("_iris_setosa_func(X['Petal.Length'])", function_defs = [_iris_setosa_func]),
		"versicolor" : Expression("_iris_versicolor_func(X['Petal.Length'], X['Petal.Width'])", function_defs = [_iris_versicolor_func]),
		"virginica" : Expression("_iris_virginica_func(X['Petal.Length'], X['Petal.Width'])", function_defs = [_iris_virginica_func])
	}

	pipeline = PMMLPipeline([
		("classifier", ExpressionClassifier(class_exprs, normalization_method = "simplemax"))
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

def build_mlp_iris(iris_df, name, transformer, predict_proba_transformer = None):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = PMMLPipeline([
		("decorator", ContinuousDomain()),
		("transformer", transformer),
		("classifier", LogisticRegression(random_state = 13))
	], predict_proba_transformer = predict_proba_transformer)
	pipeline.fit(iris_X, iris_y)
	pipeline.verify(iris_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	if predict_proba_transformer is not None:
		species_proba = DataFrame(pipeline.predict_proba_transform(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)", "decision"])
	else:
		species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

def build_pca_iris(iris_df, name):
	iris_X, iris_y = split_csv(iris_df)

	pca_names = ["pca_1", "pca_2"]

	memory = Memory()

	pca_union = FeatureUnion([
		("first", ExpressionTransformer("X[0]")),
		("second", ExpressionTransformer("X[1]")),
		("ratio", Alias(ExpressionTransformer("X[0] / X[1]"), name = "pca_ratio", prefit = True))
	])

	pipeline = PMMLPipeline([
		("pca", make_pipeline(MultiAlias(PCA(n_components = len(pca_names)), names = pca_names), make_memorizer_union(memory = memory, names = pca_names))),
		("classifier", LogisticRegression())
	], predict_transformer = make_pipeline(make_recaller_union(memory = memory, names = pca_names), pca_union))
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict_transform(iris_X), columns = ["Species", "xref({})".format(pca_names[0]), "xref({})".format(pca_names[1]), "pca_ratio"])
	store_csv(species, name)

def build_ruleset_iris(iris_df, name):
	iris_X, iris_y = split_csv(iris_df)

	classifier = RuleSetClassifier([
		(Predicate("X['Petal.Length'] >= 2.45 and X['Petal.Width'] < 1.75"), "versicolor"),
		("X['Petal.Length'] >= 2.45", "virginica")
	], default_score = "setosa")
	pipeline = PMMLPipeline([
		("domain", ContinuousDomain(display_name = ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"])),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	store_csv(species, name)

def build_selectfirst_iris(iris_df, name):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = PMMLPipeline([
		("mapper", DataFrameMapper([
			(iris_X.columns.values, ContinuousDomain())
		])),
		("classifier", SelectFirstClassifier([
			("select", Pipeline([
				("classifier", DecisionTreeClassifier(random_state = 13))
			]), "X[1] <= 3"),
			("default", Pipeline([
				("scaler", StandardScaler()),
				("classifier", LogisticRegression(multi_class = "ovr", solver = "liblinear"))
			]), str(True))
		]))
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_chaid_iris(iris_df, "CHAIDIris")
	build_expr_iris(iris_df, "ExpressionIris")

	mlp = MLPRegressor(hidden_layer_sizes = (11, ), solver = "lbfgs", random_state = 13)

	build_mlp_iris(iris_df, "MLPAutoencoderIris", MLPTransformer(mlp), predict_proba_transformer = Alias(BusinessDecisionTransformer("'yes' if X[1] >= 0.95 else 'no'", "Is the predicted species definitely versicolor?", [("yes", "Is versicolor"), ("no", "Is not versicolor")], prefit = True), "decision", prefit = True))
	build_mlp_iris(iris_df, "MLPTransformerIris", MLPTransformer(mlp, transformer_output_layer = 1))

	build_pca_iris(iris_df, "PCAIris")

	build_ruleset_iris(iris_df, "RuleSetIris")

	build_selectfirst_iris(iris_df, "SelectFirstIris")

def build_chaid_auto(auto_df, name):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

	auto_X[cat_cols] = auto_X[cat_cols].astype("Int64")

	bins = make_bins(auto_X, cont_cols)
	labels = make_bin_labels(bins)

	pipeline = PMMLPipeline([
		("mapper", make_chaid_dataframe_mapper(cont_cols, cat_cols, bins, labels)),
		("regressor", CHAIDRegressor(config = {"max_depth" : 7, "min_child_node_size" : 10}))
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	node_id = Series(pipeline._final_estimator.apply(auto_X), dtype = int, name = "nodeId")
	store_csv(node_id, name)

def _scale_displacement(displacement):
	return (displacement - 194.41) / 104.51

def _scale_weight(weight):
	return (weight - 2977.58) / 848.32

def build_expr_auto(auto_df, name):
	auto_X, auto_y = split_csv(auto_df)

	expr = Expression("-1.724 * _scale_displacement(X['displacement']) + 4.879 * _scale_weight(X['weight']) + 23.45", function_defs = [_scale_displacement, _scale_weight])

	pipeline = PMMLPipeline([
		("regressor", ExpressionRegressor(expr, normalization_method = "none"))
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = Series(pipeline.predict(auto_X), name = "mpg")
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	# XXX
	auto_df["cylinders"] = auto_df["cylinders"].astype(int)
	auto_df["model_year"] = auto_df["model_year"].astype(int)
	auto_df["origin"] = auto_df["origin"].astype(int)

	build_auto(auto_df, GBDTLMRegressor(RandomForestRegressor(n_estimators = 7, max_depth = 6, random_state = 13), LinearRegression()), "GBDTLMAuto")

	auto_df = load_auto("Auto")

	build_chaid_auto(auto_df, "CHAIDAuto")
	build_expr_auto(auto_df, "ExpressionAuto")

	build_multi_auto(auto_df, EstimatorChain([("acceleration", Link(DecisionTreeRegressor(max_depth = 3, random_state = 13), augment_funcs = ["predict", "apply"]), str(True)), ("mpg", LinearRegression(), str(True))]), "MultiEstimatorChainAuto")

	auto_df = load_auto("AutoNA")

	build_chaid_auto(auto_df, "CHAIDAutoNA")

if "Housing" in datasets:
	housing_df = load_housing("Housing")

	build_housing(housing_df, GBDTLMRegressor(GradientBoostingRegressor(n_estimators = 31, random_state = 13), LinearRegression()), "GBDTLMHousing")

def build_wine(wine_df, regressor, name, eval_rows = True):
	wine_X, wine_y = split_csv(wine_df)

	cols = wine_X.columns.values.tolist()

	memory = Memory()

	memorizer = Memorizer(memory, ["subset"])
	recaller = Recaller(memory, ["subset"])

	regressor.controller = recaller

	def make_scaler(name):
		red_scaler = Alias(StandardScaler(), name = "standardScaler({}, red)".format(name))
		white_scaler = Alias(StandardScaler(), name = "standardScaler({}, white)".format(name))

		return SelectFirstTransformer([
			("red", red_scaler, "X[0] == 'red'" if eval_rows else "X[:, 0] == 'red'"),
			("white", white_scaler, "X[0] == 'white'" if eval_rows else "X[:, 0] == 'white'")
		], controller = recaller, eval_rows = eval_rows)

	pipeline = PMMLPipeline([
		("mapper", ColumnTransformer([
			("cont", Pipeline([("domain", ContinuousDomain())]), cols[0:-1]),
			("cat", Pipeline([("domain", CategoricalDomain()), ("memorizer", memorizer)]), cols[-1:])
		])),
		("scaler", ColumnTransformer(
			[(cols[idx], make_scaler(cols[idx]), [idx]) for idx in range(0, len(cols) - 1)]
		)),
		("regressor", regressor)
	])
	pipeline.fit(wine_X, wine_y)
	store_pkl(pipeline, name)
	quality = Series(pipeline.predict(wine_X), name = "quality")
	store_csv(quality, name)

if "Wine" in datasets:
	wine_df = load_wine("Wine")

	def make_steps(eval_rows):
		return [
			("red", LinearRegression(), "X[0] == 'red'" if eval_rows else "X[:, 0] == 'red'"),
			("white", LinearRegression(), "X[0] == 'white'" if eval_rows else "X[:, 0] == 'white'")
		]

	build_wine(wine_df, EstimatorChain(make_steps(eval_rows = True), multioutput = False), "EstimatorChainWine")
	build_wine(wine_df, SelectFirstRegressor(make_steps(eval_rows = False), eval_rows = False), "SelectFirstWine", eval_rows = False)

def build_airline(airline_df, regressor, name):
	lags = [1, 2, 3]
	max_lag = max(lags)

	union = FeatureUnion(
		[("lag_{}".format(lag), LagTransformer(n = lag)) for lag in lags] +
		[("rolling_average_3", RollingAggregateTransformer(function = "avg", n = 12))] +
		[("rolling_sum_3", RollingAggregateTransformer(function = "sum", n = 12))]
	)

	X = airline_df[["Passengers"]]
	y = airline_df["Passengers"]

	Xt = union.transform(X)

	regressor.fit(Xt[max_lag:], y[max_lag:])

	yt = numpy.full_like(y, fill_value = numpy.nan, dtype = float)
	yt[max_lag:] = regressor.predict(Xt[max_lag:])

	pipeline = PMMLPipeline([
		("union", union),
		("regressor", regressor)
	])
	pipeline.active_fields = X.columns.values
	store_pkl(pipeline, name)
	passengers = Series(yt, name = "y")
	store_csv(passengers, name)

if "Airline" in datasets:
	airline_df = load_airline("Airline")

	build_airline(airline_df, LinearRegression(), "LinearRegressionAirline")
