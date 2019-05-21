from common import *

from h2o import H2OFrame
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingClassifier, BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNetCV, HuberRegressor, LarsCV, LassoCV, LassoLarsCV, LinearRegression, LogisticRegression, LogisticRegressionCV, OrthogonalMatchingPursuitCV, RidgeCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Binarizer, FunctionTransformer, Imputer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from sklearn2pmml import SelectorProxy
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain, MultiDomain
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn2pmml.feature_selection import SelectUnique
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import Aggregator, ConcatTransformer, CutTransformer, ExpressionTransformer, LookupTransformer, MatchesTransformer, MultiLookupTransformer, PMMLLabelBinarizer, PMMLLabelEncoder, PowerFunctionTransformer, ReplaceTransformer, SubstringTransformer, StringNormalizer
from sklearn2pmml.preprocessing.h2o import H2OFrameCreator
from sklearn2pmml.ruleset import RuleSetClassifier
from sklearn_pandas import CategoricalImputer, DataFrameMapper
from xgboost.sklearn import XGBClassifier, XGBRegressor

import h2o
import numpy
import pandas
import sys

def pipeline_transform(pipeline, X):
	identity_pipeline = Pipeline(pipeline.steps[: -1] + [("estimator", None)])
	return identity_pipeline._transform(X)

def make_interaction(left, right):
	pipeline = Pipeline([
		("mapper", DataFrameMapper([
			([left], LabelBinarizer()),
			([right], LabelBinarizer())
		])),
		("polynomial", PolynomialFeatures())
	])
	return pipeline

class OptimalLGBMClassifier(LGBMClassifier):

	def __init__(self, objective, n_estimators, num_iteration = 0, random_state = 13, n_jobs = -1):
		super(OptimalLGBMClassifier, self).__init__(objective = objective, n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs)
		self.num_iteration = num_iteration

	def predict(self, X, raw_score = False, num_iteration = 0, pred_leaf = False, pred_contrib = False):
		return super(OptimalLGBMClassifier, self).predict(X = X, raw_score = raw_score, num_iteration = self.num_iteration, pred_leaf = pred_leaf, pred_contrib = pred_contrib)

	def predict_proba(self, X, raw_score = False, num_iteration = 0, pred_leaf = False, pred_contrib = False):
		return super(OptimalLGBMClassifier, self).predict_proba(X = X, raw_score = raw_score, num_iteration = self.num_iteration, pred_leaf = pred_leaf, pred_contrib = pred_contrib)

class OptimalLGBMRegressor(LGBMRegressor):

	def __init__(self, objective, n_estimators, num_iteration = 0, random_state = 13, n_jobs = -1):
		super(OptimalLGBMRegressor, self).__init__(objective = objective, n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs)
		self.num_iteration = num_iteration

	def predict(self, X, raw_score = False, num_iteration = 0, pred_leaf = False, pred_contrib = False):
		return super(OptimalLGBMRegressor, self).predict(X = X, raw_score = raw_score, num_iteration = self.num_iteration, pred_leaf = pred_leaf, pred_contrib = pred_contrib)

class OptimalXGBClassifier(XGBClassifier):

	def __init__(self, objective, ntree_limit = 0, n_jobs = 1, random_state = 0, missing = None):
		super(OptimalXGBClassifier, self).__init__(objective = objective, n_jobs = n_jobs, random_state = random_state, missing = missing)
		self.ntree_limit = ntree_limit

	def predict(self, data, ntree_limit = 0):
		return super(OptimalXGBClassifier, self).predict(data = data, ntree_limit = self.ntree_limit)

	def predict_proba(self, data, ntree_limit = 0):
		return super(OptimalXGBClassifier, self).predict_proba(data = data, ntree_limit = self.ntree_limit)

class OptimalXGBRegressor(XGBRegressor):

	def __init__(self, objective, ntree_limit = 0, n_jobs = 1, random_state = 0, missing = None):
		super(OptimalXGBRegressor, self).__init__(objective = objective, n_jobs = n_jobs, random_state = random_state, missing = missing)
		self.ntree_limit = ntree_limit

	def predict(self, data, ntree_limit = 0):
		return super(OptimalXGBRegressor, self).predict(data = data, ntree_limit = self.ntree_limit)

datasets = "Audit,Auto,Housing,Iris,Sentiment,Versicolor,Wheat"

with_h2o = False

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = sys.argv[1]
	if len(sys.argv) > 2:
		with_h2o = "H2O" in sys.argv[2]

datasets = datasets.split(",")

if with_h2o:
	h2o.init()
	h2o.connect()

#
# Clustering
#

wheat_X, wheat_y = load_wheat("Wheat")

def kmeans_distance(kmeans, center, X):
	return numpy.sum(numpy.power(kmeans.cluster_centers_[center] - X, 2), axis = 1)

def build_wheat(kmeans, name, with_affinity = True, **pmml_options):
	mapper = DataFrameMapper([
		(wheat_X.columns.values, ContinuousDomain())
	])
	scaler = ColumnTransformer([
		("robust", RobustScaler(), [0, 5])
	], remainder = MinMaxScaler())
	pipeline = Pipeline([
		("mapper", mapper),
		("scaler", scaler),
		("clusterer", kmeans)
	])
	pipeline.fit(wheat_X)
	pipeline = make_pmml_pipeline(pipeline, wheat_X.columns.values)
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)
	cluster = DataFrame(pipeline.predict(wheat_X), columns = ["Cluster"])
	if with_affinity == True:
		Xt = pipeline_transform(pipeline, wheat_X)
		affinity_0 = kmeans_distance(kmeans, 0, Xt)
		affinity_1 = kmeans_distance(kmeans, 1, Xt)
		affinity_2 = kmeans_distance(kmeans, 2, Xt)
		cluster_affinity = DataFrame(numpy.transpose([affinity_0, affinity_1, affinity_2]), columns = ["affinity(0)", "affinity(1)", "affinity(2)"])
		cluster = pandas.concat((cluster, cluster_affinity), axis = 1)
	store_csv(cluster, name)

if "Wheat" in datasets:
	build_wheat(KMeans(n_clusters = 3, random_state = 13), "KMeansWheat")
	build_wheat(MiniBatchKMeans(n_clusters = 3, compute_labels = False, random_state = 13), "MiniBatchKMeansWheat")

#
# Binary classification
#

audit_X, audit_y = load_audit("Audit", stringify = False)

def build_audit(classifier, name, with_proba = True, **pmml_options):
	continuous_mapper = DataFrameMapper([
		(["Age", "Income", "Hours"], MultiDomain([ContinuousDomain() for i in range(0, 3)]))
	])
	categorical_mapper = DataFrameMapper([
		(["Employment"], [CategoricalDomain(), SubstringTransformer(0, 3), LabelBinarizer(), SelectFromModel(DecisionTreeClassifier(random_state = 13))]),
		(["Education"], [CategoricalDomain(), ReplaceTransformer("[aeiou]", ""), LabelBinarizer(), SelectFromModel(RandomForestClassifier(random_state = 13, n_estimators = 3), threshold = "1.25 * mean")]),
		(["Marital"], [CategoricalDomain(), LabelBinarizer(neg_label = -1, pos_label = 1), SelectKBest(k = 3)]),
		(["Occupation"], [CategoricalDomain(), LabelBinarizer(), SelectKBest(k = 3)]),
		(["Gender"], [CategoricalDomain(), MatchesTransformer("^Male$")]),
		(["Deductions"], [CategoricalDomain()]),
	])
	pipeline = Pipeline([
		("union", FeatureUnion([
			("continuous", continuous_mapper),
			("categorical", Pipeline([
				("mapper", categorical_mapper),
				("polynomial", PolynomialFeatures())
			]))
		])),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	pipeline = make_pmml_pipeline(pipeline, audit_X.columns.values, audit_y.name)
	pipeline.configure(**pmml_options)
	if isinstance(classifier, XGBClassifier):
		pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	if with_proba == True:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	build_audit(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 2), "DecisionTreeAudit", compact = False)
	build_audit(BaggingClassifier(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleAudit")
	build_audit(DummyClassifier(strategy = "most_frequent"), "DummyAudit")
	build_audit(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesAudit")
	build_audit(GradientBoostingClassifier(random_state = 13, loss = "exponential", init = None), "GradientBoostingAudit")
	build_audit(OptimalLGBMClassifier(objective = "binary", n_estimators = 37, num_iteration = 17), "LGBMAudit", num_iteration = 17)
	build_audit(LinearDiscriminantAnalysis(solver = "lsqr"), "LinearDiscriminantAnalysisAudit")
	build_audit(LinearSVC(penalty = "l1", dual = False, random_state = 13), "LinearSVCAudit", with_proba = False)
	build_audit(LogisticRegression(multi_class = "multinomial", solver = "newton-cg", max_iter = 500), "MultinomialLogisticRegressionAudit")
	build_audit(LogisticRegressionCV(multi_class = "ovr"), "OvRLogisticRegressionAudit")
	build_audit(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleAudit")
	build_audit(GaussianNB(), "NaiveBayesAudit")
	build_audit(OneVsRestClassifier(LogisticRegression()), "OneVsRestAudit")
	build_audit(RandomForestClassifier(random_state = 13, min_samples_leaf = 3), "RandomForestAudit", flat = True)
	build_audit(RidgeClassifierCV(), "RidgeAudit", with_proba = False)
	build_audit(BaggingClassifier(RidgeClassifier(random_state = 13), random_state = 13, n_estimators = 3, max_features = 0.5), "RidgeEnsembleAudit")
	build_audit(SVC(), "SVCAudit", with_proba = False)
	build_audit(VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression())], voting = "soft", weights = [3, 1, 2]), "VotingEnsembleAudit")
	build_audit(OptimalXGBClassifier(objective = "binary:logistic", ntree_limit = 71, random_state = 13), "XGBAudit", byte_order = "LITTLE_ENDIAN", charset = "US-ASCII", ntree_limit = 71)

audit_X, audit_y = load_audit("Audit")

def build_audit_cat(classifier, name, with_proba = True, **fit_params):
	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["Age", "Income"]] +
		[(["Hours"], [ContinuousDomain(), CutTransformer(bins = [0, 20, 40, 60, 80, 100], labels = False, right = False, include_lowest = True)])] +
		[(["Employment", "Education", "Marital", "Occupation"], [MultiDomain([CategoricalDomain(), CategoricalDomain(), CategoricalDomain(), CategoricalDomain()]), OrdinalEncoder()])] +
		[([column], [CategoricalDomain(), LabelEncoder()]) for column in ["Gender", "Deductions"]]
	)
	pipeline = Pipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y, **fit_params)
	pipeline = make_pmml_pipeline(pipeline, audit_X.columns.values, audit_y.name)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	if with_proba == True:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	build_audit_cat(LGBMClassifier(objective = "binary", n_estimators = 37), "LGBMAuditCat", classifier__categorical_feature = [2, 3, 4, 5, 6, 7, 8])

def build_audit_h2o(classifier, name):
	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["Age", "Hours", "Income"]] +
		[([column], CategoricalDomain()) for column in ["Employment", "Education", "Marital", "Occupation", "Gender", "Deductions"]]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("uploader", H2OFrameCreator()),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, H2OFrame(audit_y.to_frame(), column_types = ["categorical"]))
	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	classifier = pipeline._final_estimator
	store_mojo(classifier, name)
	store_pkl(pipeline, name)
	adjusted = pipeline.predict(audit_X)
	adjusted.set_names(["h2o(Adjusted)", "probability(0)", "probability(1)"])
	store_csv(adjusted.as_data_frame(), name)

if "Audit" in datasets and with_h2o:
	build_audit_h2o(H2OGradientBoostingEstimator(distribution = "bernoulli", ntrees = 17), "H2OGradientBoostingAudit")
	build_audit_h2o(H2OGeneralizedLinearEstimator(family = "binomial"), "H2OLogisticRegressionAudit")
	build_audit_h2o(H2ORandomForestEstimator(distribution = "bernoulli", seed = 13), "H2ORandomForestAudit")

audit_dict_X = audit_X.to_dict("records")

def build_audit_dict(classifier, name, with_proba = True):
	pipeline = PMMLPipeline([
		("dict-transformer", DictVectorizer()),
		("classifier", classifier)
	])
	pipeline.fit(audit_dict_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_dict_X), columns = ["Adjusted"])
	if with_proba == True:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_dict_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	build_audit_dict(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeAuditDict")
	build_audit_dict(LogisticRegression(), "LogisticRegressionAuditDict")

audit_na_X, audit_na_y = load_audit("AuditNA")

def build_audit_na(classifier, name, with_proba = True, predict_transformer = None, predict_proba_transformer = None, apply_transformer = None, **pmml_options):
	employment_mapping = {
		"CONSULTANT" : "PRIVATE",
		"PSFEDERAL" : "PUBLIC",
		"PSLOCAL" : "PUBLIC",
		"PSSTATE" : "PUBLIC",
		"SELFEMP" : "PRIVATE",
		"PRIVATE" : "PRIVATE"
	}
	gender_mapping = {
		"FEMALE" : 0.0,
		"MALE" : 1.0,
		"MISSING_VALUE" : 0.5
	}
	mapper = DataFrameMapper(
		[(["Age"], [ContinuousDomain(missing_values = None, with_data = False), Alias(ExpressionTransformer("X[0] if pandas.notnull(X[0]) else -999"), name = "flag_missing(Age, -999)"), Imputer(missing_values = -999)])] +
		[(["Age"], MissingIndicator())] +
		[(["Hours"], [ContinuousDomain(missing_values = None, with_data = False), Alias(ExpressionTransformer("-999 if pandas.isnull(X[0]) else X[0]"), name = "flag_missing(Hours, -999)"), SimpleImputer(missing_values = -999, add_indicator = True)])] +
		[(["Income"], [ContinuousDomain(missing_values = None, outlier_treatment = "as_missing_values", low_value = 5000, high_value = 200000, with_data = False), SimpleImputer(strategy = "median", add_indicator = True)])] +
		[(["Employment"], [CategoricalDomain(missing_values = None, with_data = False), CategoricalImputer(missing_values = None), StringNormalizer(function = "uppercase"), LookupTransformer(employment_mapping, "OTHER"), StringNormalizer(function = "lowercase"), PMMLLabelBinarizer()])] +
		[([column], [CategoricalDomain(missing_values = None, missing_value_replacement = "N/A", with_data = False), SimpleImputer(missing_values = "N/A", strategy = "most_frequent"), StringNormalizer(function = "lowercase"), PMMLLabelBinarizer()]) for column in ["Education", "Marital", "Occupation"]] +
		[(["Gender"], [CategoricalDomain(missing_values = None, with_data = False), SimpleImputer(strategy = "constant"), StringNormalizer(function = "uppercase"), LookupTransformer(gender_mapping, None)])]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	], predict_transformer = predict_transformer, predict_proba_transformer = predict_proba_transformer, apply_transformer = apply_transformer)
	pipeline.fit(audit_na_X, audit_na_y)
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_na_X), columns = ["Adjusted"])
	if with_proba == True:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_na_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	if isinstance(classifier, DecisionTreeClassifier):
		Xt = pipeline_transform(pipeline, audit_na_X)
		adjusted_apply = DataFrame(classifier.apply(Xt), columns = ["nodeId"])
		adjusted = pandas.concat((adjusted, adjusted_apply), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	build_audit_na(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeAuditNA", apply_transformer = Alias(ExpressionTransformer("X[0] - 1"), "eval(nodeId)", prefit = True), winner_id = True, class_extensions = {"event" : {"0" : False, "1" : True}})
	build_audit_na(LogisticRegression(solver = "newton-cg", max_iter = 500), "LogisticRegressionAuditNA", predict_proba_transformer = Alias(ExpressionTransformer("1 if X[1] > 0.75 else 0"), name = "eval(probability(1))", prefit = True))
	build_audit_na(OptimalXGBClassifier(objective = "binary:logistic", ntree_limit = 11, random_state = 13), "XGBAuditNA", predict_transformer = Alias(ExpressionTransformer("X[0]"), name = "eval(Adjusted)", prefit = True))

versicolor_X, versicolor_y = load_versicolor("Versicolor")

def build_versicolor(classifier, name, with_proba = True, **pmml_options):
	transformer = ColumnTransformer([
		("continuous_columns", Pipeline([
			("domain", ContinuousDomain()),
			("scaler", RobustScaler())
		]), versicolor_X.columns.values)
	])
	pipeline = Pipeline([
		("transformer", transformer),
		("transformer-selector-pipeline", Pipeline([
			("polynomial", PolynomialFeatures(degree = 3)),
			("selector", SelectKBest(k = "all"))
		])),
		("classifier", classifier)
	])
	pipeline.fit(versicolor_X, versicolor_y)
	pipeline = make_pmml_pipeline(pipeline, versicolor_X.columns.values, versicolor_y.name)
	pipeline.configure(**pmml_options)
	pipeline.verify(versicolor_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	if with_proba == True:
		species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Versicolor" in datasets:
	build_versicolor(DummyClassifier(strategy = "prior"), "DummyVersicolor")
	build_versicolor(KNeighborsClassifier(), "KNNVersicolor", with_proba = False)
	build_versicolor(MLPClassifier(activation = "tanh", hidden_layer_sizes = (8,), solver = "lbfgs", random_state = 13, tol = 0.1, max_iter = 100), 	"MLPVersicolor")
	build_versicolor(SGDClassifier(random_state = 13, max_iter = 100), "SGDVersicolor", with_proba = False)
	build_versicolor(SGDClassifier(random_state = 13, loss = "log", max_iter = 100), "SGDLogVersicolor")
	build_versicolor(SVC(), "SVCVersicolor", with_proba = False)
	build_versicolor(NuSVC(), "NuSVCVersicolor", with_proba = False)

versicolor_X, versicolor_y = load_versicolor("Versicolor")

def build_versicolor_direct(classifier, name, with_proba = True, **pmml_options):
	transformer = ColumnTransformer([
		("all", "passthrough", ["Petal.Length", "Petal.Width"])
	], remainder = "drop")
	pipeline = PMMLPipeline([
		("transformer", transformer),
		("classifier", classifier)
	])
	pipeline.fit(versicolor_X, versicolor_y)
	pipeline.configure(**pmml_options)
	pipeline.verify(versicolor_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	if with_proba == True:
		species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Versicolor" in datasets:
	build_versicolor_direct(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeVersicolor", compact = False)

#
# Multi-class classification
#

iris_X, iris_y = load_iris("Iris")

def build_iris(classifier, name, with_proba = True, **pmml_options):
	pipeline = Pipeline([
		("pipeline", Pipeline([
			("mapper", DataFrameMapper([
				(iris_X.columns.values, ContinuousDomain()),
				(["Sepal.Length", "Petal.Length"], Aggregator(function = "mean")),
				(["Sepal.Width", "Petal.Width"], Aggregator(function = "mean"))
			])),
			("transform", FeatureUnion([
				("normal_scale", FunctionTransformer(None)),
				("log_scale", FunctionTransformer(numpy.log10)),
				("power_scale", PowerFunctionTransformer(power = 2))
			]))
		])),
		("pca", IncrementalPCA(n_components = 3, whiten = True)),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	pipeline = make_pmml_pipeline(pipeline, iris_X.columns.values, iris_y.name)
	pipeline.configure(**pmml_options)
	if isinstance(classifier, XGBClassifier):
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	if with_proba == True:
		species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	build_iris(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeIris", compact = False)
	build_iris(BaggingClassifier(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleIris")
	build_iris(DummyClassifier(strategy = "constant", constant = "versicolor"), "DummyIris")
	build_iris(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesIris")
	build_iris(GradientBoostingClassifier(random_state = 13, init = None, n_estimators = 17), "GradientBoostingIris")
	build_iris(KNeighborsClassifier(), "KNNIris", with_proba = False)
	build_iris(OptimalLGBMClassifier(objective = "multiclass", n_estimators = 7, num_iteration = 3), "LGBMIris", num_iteration = 3)
	build_iris(LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysisIris")
	build_iris(LinearSVC(random_state = 13), "LinearSVCIris", with_proba = False)
	build_iris(LogisticRegression(multi_class = "multinomial", solver = "lbfgs"), "MultinomialLogisticRegressionIris")
	build_iris(LogisticRegressionCV(multi_class = "ovr"), "OvRLogisticRegressionIris")
	build_iris(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleIris")
	build_iris(MLPClassifier(hidden_layer_sizes = (6,), solver = "lbfgs", random_state = 13, tol = 0.1, max_iter = 100), "MLPIris")
	build_iris(GaussianNB(), "NaiveBayesIris")
	build_iris(OneVsRestClassifier(LogisticRegression()), "OneVsRestIris")
	build_iris(RandomForestClassifier(random_state = 13, min_samples_leaf = 5), "RandomForestIris", flat = True)
	build_iris(RidgeClassifierCV(), "RidgeIris", with_proba = False)
	build_iris(BaggingClassifier(RidgeClassifier(random_state = 13), random_state = 13, n_estimators = 3, max_features = 0.5), "RidgeEnsembleIris")
	build_iris(SGDClassifier(random_state = 13, max_iter = 100), "SGDIris", with_proba = False)
	build_iris(SGDClassifier(random_state = 13, loss = "log", max_iter = 100), "SGDLogIris")
	build_iris(SVC(), "SVCIris", with_proba = False)
	build_iris(NuSVC(), "NuSVCIris", with_proba = False)
	build_iris(VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression())]), "VotingEnsembleIris", with_proba = False)
	build_iris(OptimalXGBClassifier(objective = "multi:softprob", ntree_limit = 7), "XGBIris", ntree_limit = 7)

if "Iris" in datasets:
	classifier = RuleSetClassifier([
		("X['Petal.Length'] >= 2.45 and X['Petal.Width'] < 1.75", "versicolor"),
		("X['Petal.Length'] >= 2.45", "virginica")
	], default_score = "setosa")
	pipeline = PMMLPipeline([
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, "RuleSetIris")
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	store_csv(species, "RuleSetIris")

#
# Text classification
#

sentiment_X, sentiment_y = load_sentiment("Sentiment")

def build_sentiment(classifier, name, with_proba = True, **pmml_options):
	pipeline = PMMLPipeline([
		("tf-idf", TfidfVectorizer(analyzer = "word", preprocessor = None, strip_accents = None, lowercase = True, token_pattern = None, tokenizer = Splitter(), stop_words = "english", ngram_range = (1, 2), norm = None, dtype = (numpy.float32 if isinstance(classifier, RandomForestClassifier) else numpy.float64))),
		("selector", SelectKBest(f_classif, k = 500)),
		("classifier", classifier)
	])
	pipeline.fit(sentiment_X, sentiment_y)
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)
	score = DataFrame(pipeline.predict(sentiment_X), columns = ["Score"])
	if with_proba == True:
		score_proba = DataFrame(pipeline.predict_proba(sentiment_X), columns = ["probability(0)", "probability(1)"])
		score = pandas.concat((score, score_proba), axis = 1)
	store_csv(score, name)

if "Sentiment" in datasets:
	build_sentiment(LinearSVC(random_state = 13), "LinearSVCSentiment", with_proba = False)
	build_sentiment(LogisticRegressionCV(), "LogisticRegressionSentiment")
	build_sentiment(RandomForestClassifier(random_state = 13, min_samples_leaf = 3), "RandomForestSentiment", compact = False)

#
# Regression
#

auto_X, auto_y = load_auto("Auto")

auto_X["model_year"] = auto_X["model_year"].astype(int)
auto_X["origin"] = auto_X["origin"].astype(int)

def build_auto(regressor, name, **pmml_options):
	cylinders_origin_mapping = {
		(8, 1) : "8/1",
		(6, 1) : "6/1",
		(4, 1) : "4/1",
		(6, 2) : "6/2",
		(4, 2) : "4/2",
		(4, 3) : "4/3"
	}
	mapper = DataFrameMapper([
		(["cylinders", "origin"], [MultiDomain([CategoricalDomain(), CategoricalDomain()]), MultiLookupTransformer(cylinders_origin_mapping, default_value = "other"), LabelBinarizer()]),
		(["cylinders"], Alias(ExpressionTransformer("X[0] % 2.0 > 0.0"), name = "odd(cylinders)", prefit = True)),
		(["model_year"], [CategoricalDomain(), Binarizer(threshold = 77)], {"alias" : "bin(model_year, 77)"}), # Pre/post 1973 oil crisis effects
		(["model_year", "origin"], [MultiDomain([CategoricalDomain(), CategoricalDomain()]), ConcatTransformer("/"), LabelBinarizer(), SelectorProxy(SelectFromModel(RandomForestRegressor(random_state = 13, n_estimators = 3), threshold = "1.25 * mean"))]),
		(["displacement", "horsepower", "weight", "acceleration"], [ContinuousDomain(), StandardScaler()]),
		(["weight", "displacement"], ExpressionTransformer("(X[0] / X[1]) + 0.5"), {"alias" : "weight / displacement + 0.5"})
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("selector", SelectUnique()),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	pipeline.configure(**pmml_options)
	if isinstance(regressor, XGBRegressor):
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	build_auto(AdaBoostRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 17), "AdaBoostAuto")
	build_auto(ARDRegression(normalize = True), "BayesianARDAuto")
	build_auto(BayesianRidge(normalize = True), "BayesianRidgeAuto")
	build_auto(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 2), "DecisionTreeAuto", compact = False)
	build_auto(BaggingRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleAuto")
	build_auto(DummyRegressor(strategy = "median"), "DummyAuto")
	build_auto(ElasticNetCV(random_state = 13), "ElasticNetAuto")
	build_auto(ExtraTreesRegressor(random_state = 13, min_samples_leaf = 5), "ExtraTreesAuto")
	build_auto(GradientBoostingRegressor(random_state = 13, init = None), "GradientBoostingAuto")
	build_auto(HuberRegressor(), "HuberAuto")
	build_auto(LarsCV(), "LarsAuto")
	build_auto(LassoCV(random_state = 13), "LassoAuto")
	build_auto(LassoLarsCV(), "LassoLarsAuto")
	build_auto(OptimalLGBMRegressor(objective = "regression", n_estimators = 17, num_iteration = 11), "LGBMAuto", num_iteration = 11)
	build_auto(LinearRegression(), "LinearRegressionAuto")
	build_auto(BaggingRegressor(LinearRegression(), random_state = 13, max_features = 0.75), "LinearRegressionEnsembleAuto")
	build_auto(OrthogonalMatchingPursuitCV(), "OMPAuto")
	build_auto(RandomForestRegressor(random_state = 13, min_samples_leaf = 3), "RandomForestAuto", flat = True)
	build_auto(RidgeCV(), "RidgeAuto")
	build_auto(TheilSenRegressor(n_subsamples = 31, random_state = 13), "TheilSenAuto")
	build_auto(VotingRegressor([("dt", DecisionTreeRegressor(random_state = 13)), ("knn", KNeighborsRegressor()), ("lr", LinearRegression())], weights = [3, 1, 2]), "VotingEnsembleAuto")
	build_auto(OptimalXGBRegressor(objective = "reg:linear", ntree_limit = 31), "XGBAuto", ntree_limit = 31)

if "Auto" in datasets:
	build_auto(TransformedTargetRegressor(DecisionTreeRegressor(random_state = 13)), "TransformedDecisionTreeAuto")
	build_auto(TransformedTargetRegressor(LinearRegression(), func = numpy.log, inverse_func = numpy.exp), "TransformedLinearRegressionAuto")

def build_auto_h2o(regressor, name):
	transformer = ColumnTransformer(
		[(column, CategoricalDomain(), [column]) for column in ["cylinders", "model_year", "origin"]] +
		[(column, ContinuousDomain(), [column]) for column in ["displacement", "horsepower", "weight", "acceleration"]]
	)
	pipeline = PMMLPipeline([
		("transformer", transformer),
		("uploader", H2OFrameCreator(column_names = ["cylinders", "model_year", "origin", "displacement", "horsepower", "weight", "acceleration"], column_types = ["enum", "enum", "enum", "numeric", "numeric", "numeric", "numeric"])),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, H2OFrame(auto_y.to_frame()))
	pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	regressor = pipeline._final_estimator
	store_mojo(regressor, name)
	store_pkl(pipeline, name)
	mpg = pipeline.predict(auto_X)
	mpg.set_names(["mpg"])
	store_csv(mpg.as_data_frame(), name)

if "Auto" in datasets and with_h2o:
	build_auto_h2o(H2OGradientBoostingEstimator(distribution = "gaussian", ntrees = 17), "H2OGradientBoostingAuto")
	build_auto_h2o(H2OGeneralizedLinearEstimator(family = "gaussian"), "H2OLinearRegressionAuto")
	build_auto_h2o(H2ORandomForestEstimator(distribution = "gaussian", seed = 13), "H2ORandomForestAuto")

auto_na_X, auto_na_y = load_auto("AutoNA")

auto_na_X["cylinders"] = auto_na_X["cylinders"].fillna(-1).astype(int)
auto_na_X["model_year"] = auto_na_X["model_year"].fillna(-1).astype(int)
auto_na_X["origin"] = auto_na_X["origin"].fillna(-1).astype(int)

def build_auto_na(regressor, name, predict_transformer = None, apply_transformer = None, **pmml_options):
	mapper = DataFrameMapper(
		[([column], [CategoricalDomain(missing_values = -1), CategoricalImputer(missing_values = -1), PMMLLabelBinarizer()]) for column in ["cylinders", "model_year"]] +
		[(["origin"], [CategoricalDomain(missing_values = -1), SimpleImputer(missing_values = -1, strategy = "most_frequent"), OneHotEncoder()])] +
		[(["acceleration"], [ContinuousDomain(missing_values = None), CutTransformer(bins = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25], labels = False), CategoricalImputer(), LabelBinarizer()])] +
		[(["displacement"], [ContinuousDomain(missing_values = None), Imputer(), CutTransformer(bins = [0, 100, 200, 300, 400, 500], labels = ["XS", "S", "M", "L", "XL"]), LabelBinarizer()])] +
		[(["horsepower"], [ContinuousDomain(missing_values = None, outlier_treatment = "as_extreme_values", low_value = 50, high_value = 225), SimpleImputer(strategy = "median")])] +
		[(["weight"], [ContinuousDomain(missing_values = None, outlier_treatment = "as_extreme_values", low_value = 2000, high_value = 5000), Imputer(strategy = "median")])]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	], predict_transformer = predict_transformer, apply_transformer = apply_transformer)
	pipeline.fit(auto_na_X, auto_na_y)
	if isinstance(regressor, DecisionTreeRegressor):
		tree = regressor.tree_
		node_impurity = {node_idx : tree.impurity[node_idx] for node_idx in range(0, tree.node_count) if tree.impurity[node_idx] != 0.0}
		pmml_options["node_extensions"] = {regressor.criterion : node_impurity}
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_na_X), columns = ["mpg"])
	if isinstance(regressor, DecisionTreeRegressor):
		Xt = pipeline_transform(pipeline, auto_na_X)
		mpg_apply = DataFrame(regressor.apply(Xt), columns = ["nodeId"])
		mpg = pandas.concat((mpg, mpg_apply), axis = 1)
	store_csv(mpg, name)

if "Auto" in datasets:
	build_auto_na(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 2), "DecisionTreeAutoNA", apply_transformer = Alias(ExpressionTransformer("X[0] - 1"), "eval(nodeId)", prefit = True), winner_id = True)
	build_auto_na(LinearRegression(), "LinearRegressionAutoNA", predict_transformer = CutTransformer(bins = [0, 10, 20, 30, 40], labels = ["0-10", "10-20", "20-30", "30-40"]))

housing_X, housing_y = load_housing("Housing")

def build_housing(regressor, name, with_kneighbors = False, **pmml_options):
	mapper = DataFrameMapper([
		(housing_X.columns.values, ContinuousDomain())
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("transformer-pipeline", Pipeline([
			("polynomial", PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)),
			("scaler", StandardScaler()),
			("selector", SelectPercentile(score_func = f_regression, percentile = 35)),
		])),
		("regressor", regressor)
	])
	pipeline.fit(housing_X, housing_y)
	pipeline = make_pmml_pipeline(pipeline, housing_X.columns.values, housing_y.name)
	pipeline.configure(**pmml_options)
	pipeline.verify(housing_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	medv = DataFrame(pipeline.predict(housing_X), columns = ["MEDV"])
	if with_kneighbors == True:
		Xt = pipeline_transform(pipeline, housing_X)
		kneighbors = regressor.kneighbors(Xt)
		medv_ids = DataFrame(kneighbors[1] + 1, columns = ["neighbor(" + str(x + 1) + ")" for x in range(regressor.n_neighbors)])
		medv = pandas.concat((medv, medv_ids), axis = 1)
	store_csv(medv, name)

if "Housing" in datasets:
	build_housing(AdaBoostRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 17), "AdaBoostHousing")
	build_housing(BayesianRidge(), "BayesianRidgeHousing")
	build_housing(KNeighborsRegressor(), "KNNHousing", with_kneighbors = True)
	build_housing(MLPRegressor(activation = "tanh", hidden_layer_sizes = (26,), solver = "lbfgs", random_state = 13, tol = 0.001, max_iter = 1000), "MLPHousing")
	build_housing(SGDRegressor(random_state = 13), "SGDHousing")
	build_housing(SVR(), "SVRHousing")
	build_housing(LinearSVR(random_state = 13), "LinearSVRHousing")
	build_housing(NuSVR(), "NuSVRHousing")
	build_housing(VotingRegressor([("dt", DecisionTreeRegressor(random_state = 13)), ("lr", LinearRegression())]), "VotingEnsembleHousing")

#
# Anomaly detection
#

def build_iforest_housing(iforest, name, **pmml_options):
	mapper = DataFrameMapper([
		(housing_X.columns.values, ContinuousDomain())
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("estimator", iforest)
	])
	pipeline.fit(housing_X)
	pipeline = make_pmml_pipeline(pipeline, housing_X.columns.values)
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) == -1, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name)

if "Housing" in datasets:
	build_iforest_housing(IsolationForest(random_state = 13), "IsolationForestHousing")

def build_ocsvm_housing(svm, name):
	mapper = DataFrameMapper([
		(housing_X.columns.values, ContinuousDomain())
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("scaler", MaxAbsScaler()),
		("estimator", svm)
	])
	pipeline.fit(housing_X)
	pipeline = make_pmml_pipeline(pipeline, housing_X.columns.values)
	store_pkl(pipeline, name)
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) <= 0, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name)

if "Housing" in datasets:
	build_ocsvm_housing(OneClassSVM(nu = 0.10, random_state = 13), "OneClassSVMHousing")
