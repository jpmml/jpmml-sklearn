from common import *

from pandas import DataFrame, Series
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.neural_network import MLPTransformer
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.postprocessing import BusinessDecisionTransformer
from sklearn2pmml.preprocessing import CutTransformer
from sklearn2pmml.tree.chaid import CHAIDClassifier, CHAIDRegressor

import numpy

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

audit_df = load_audit("Audit")

build_chaid_audit(audit_df, "CHAIDAudit")

audit_df = load_audit("AuditNA")

build_chaid_audit(audit_df, "CHAIDAuditNA")

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

iris_df = load_iris("Iris")

build_chaid_iris(iris_df, "CHAIDIris")

mlp = MLPRegressor(hidden_layer_sizes = (11, ), solver = "lbfgs", random_state = 13)

build_mlp_iris(iris_df, "MLPAutoencoderIris", MLPTransformer(mlp), predict_proba_transformer = Alias(BusinessDecisionTransformer("'yes' if X[1] >= 0.95 else 'no'", "Is the predicted species definitely versicolor?", [("yes", "Is versicolor"), ("no", "Is not versicolor")], prefit = True), "decision", prefit = True))
build_mlp_iris(iris_df, "MLPTransformerIris", MLPTransformer(mlp, transformer_output_layer = 1))

def build_chaid_auto(auto_df, name):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

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

auto_df = load_auto("Auto")

build_chaid_auto(auto_df, "CHAIDAuto")

auto_df = load_auto("AutoNA")

build_chaid_auto(auto_df, "CHAIDAutoNA")
