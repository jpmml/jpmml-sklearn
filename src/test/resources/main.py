from common import *

from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble.bagging import BaggingClassifier, BaggingRegressor
from sklearn.ensemble.forest import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble.iforest import IsolationForest
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ElasticNetCV, LassoCV
from sklearn.linear_model.ridge import RidgeCV, RidgeClassifier, RidgeClassifierCV
from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Binarizer, FunctionTransformer, Imputer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.svm import LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain, MultiDomain
from sklearn2pmml.feature_extraction.text import Splitter
from sklearn2pmml.preprocessing import ExpressionTransformer, LookupTransformer, MultiLookupTransformer, PMMLLabelBinarizer, PMMLLabelEncoder
from sklearn_pandas import CategoricalImputer, DataFrameMapper
from xgboost.sklearn import XGBClassifier, XGBRegressor

import numpy
import pandas

def pipeline_transform(pipeline, X):
	identity_pipeline = Pipeline(pipeline.steps[: -1] + [("estimator", None)])
	return identity_pipeline._transform(X)

def customize(estimator, **kwargs):
	for key in kwargs:
		setattr(estimator, key, kwargs[key])
	return estimator

class OptimalLGBMClassifier(LGBMClassifier):

	def __init__(self, objective, n_estimators, num_iteration = 0, random_state = 13, n_jobs = -1):
		super(OptimalLGBMClassifier, self).__init__(objective = objective, n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs)
		self.num_iteration = num_iteration

	def predict(self, X, raw_score = False, num_iteration = 0):
		return super(OptimalLGBMClassifier, self).predict(X = X, raw_score = raw_score, num_iteration = self.num_iteration)

	def predict_proba(self, X, raw_score = False, num_iteration = 0):
		return super(OptimalLGBMClassifier, self).predict_proba(X = X, raw_score = raw_score, num_iteration = self.num_iteration)

class OptimalLGBMRegressor(LGBMRegressor):

	def __init__(self, objective, n_estimators, num_iteration = 0, random_state = 13, n_jobs = -1):
		super(OptimalLGBMRegressor, self).__init__(objective = objective, n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs)
		self.num_iteration = num_iteration

	def predict(self, X, raw_score = False, num_iteration = 0):
		return super(OptimalLGBMRegressor, self).predict(X = X, raw_score = raw_score, num_iteration = self.num_iteration)

class OptimalXGBClassifier(XGBClassifier):

	def __init__(self, objective, ntree_limit = 0, missing = None):
		super(OptimalXGBClassifier, self).__init__(objective = objective, missing = missing)
		self.ntree_limit = ntree_limit

	def predict(self, data, output_margin = False, ntree_limit = 0):
		return super(OptimalXGBClassifier, self).predict(data = data, output_margin = output_margin, ntree_limit = self.ntree_limit)

	def predict_proba(self, data, output_margin = False, ntree_limit = 0):
		return super(OptimalXGBClassifier, self).predict_proba(data = data, output_margin = output_margin, ntree_limit = self.ntree_limit)

class OptimalXGBRegressor(XGBRegressor):

	def __init__(self, objective, ntree_limit = 0, missing = None):
		super(OptimalXGBRegressor, self).__init__(objective = objective, missing = missing)
		self.ntree_limit = ntree_limit

	def predict(self, data, output_margin = False, ntree_limit = 0):
		return super(OptimalXGBRegressor, self).predict(data = data, output_margin = output_margin, ntree_limit = self.ntree_limit)

#
# Clustering
#

wheat_X, wheat_y = load_wheat("Wheat.csv")

def kmeans_distance(kmeans, center, X):
	return numpy.sum(numpy.power(kmeans.cluster_centers_[center] - X, 2), axis = 1)

def build_wheat(kmeans, name, with_affinity = True, **kwargs):
	mapper = DataFrameMapper([
		(wheat_X.columns.values, ContinuousDomain())
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("scaler", MinMaxScaler()),
		("clusterer", kmeans)
	])
	pipeline.fit(wheat_X)
	pipeline = make_pmml_pipeline(pipeline, wheat_X.columns.values)
	customize(kmeans, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	cluster = DataFrame(pipeline.predict(wheat_X), columns = ["Cluster"])
	if(with_affinity == True):
		Xt = pipeline_transform(pipeline, wheat_X)
		affinity_0 = kmeans_distance(kmeans, 0, Xt)
		affinity_1 = kmeans_distance(kmeans, 1, Xt)
		affinity_2 = kmeans_distance(kmeans, 2, Xt)
		cluster_affinity = DataFrame(numpy.transpose([affinity_0, affinity_1, affinity_2]), columns = ["affinity(0)", "affinity(1)", "affinity(2)"])
		cluster = pandas.concat((cluster, cluster_affinity), axis = 1)
	store_csv(cluster, name + ".csv")

build_wheat(KMeans(n_clusters = 3, random_state = 13), "KMeansWheat")
build_wheat(MiniBatchKMeans(n_clusters = 3, compute_labels = False, random_state = 13), "MiniBatchKMeansWheat")

#
# Binary classification
#

audit_X, audit_y = load_audit("Audit.csv")

def build_audit(classifier, name, with_proba = True, **kwargs):
	continuous_mapper = DataFrameMapper([
		(["Age", "Income", "Hours"], MultiDomain([ContinuousDomain() for i in range(0, 3)]))
	])
	categorical_mapper = DataFrameMapper([
		("Employment", [CategoricalDomain(), LabelBinarizer(), SelectFromModel(DecisionTreeClassifier(random_state = 13))]),
		("Education", [CategoricalDomain(), LabelBinarizer(), SelectFromModel(RandomForestClassifier(random_state = 13, n_estimators = 3), threshold = "1.25 * mean")]),
		("Marital", [CategoricalDomain(), LabelBinarizer(neg_label = -1, pos_label = 1), SelectKBest(k = 3)]),
		("Occupation", [CategoricalDomain(), LabelBinarizer(), SelectKBest(k = 3)]),
		("Gender", [CategoricalDomain(), LabelBinarizer(neg_label = -3, pos_label = 3)]),
		("Deductions", [CategoricalDomain(), LabelEncoder()]),
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
	if isinstance(classifier, XGBClassifier):
		pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	customize(classifier, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	if(with_proba == True):
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name + ".csv")

build_audit(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 2), "DecisionTreeAudit", compact = False)
build_audit(BaggingClassifier(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleAudit")
build_audit(DummyClassifier(strategy = "most_frequent"), "DummyAudit")
build_audit(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesAudit")
build_audit(GradientBoostingClassifier(random_state = 13, loss = "exponential", init = None), "GradientBoostingAudit")
build_audit(OptimalLGBMClassifier(objective = "binary", n_estimators = 37, num_iteration = 17), "LGBMAudit")
build_audit(LinearDiscriminantAnalysis(solver = "lsqr"), "LinearDiscriminantAnalysisAudit")
build_audit(LogisticRegressionCV(), "LogisticRegressionAudit")
build_audit(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleAudit")
build_audit(GaussianNB(), "NaiveBayesAudit")
build_audit(RandomForestClassifier(random_state = 13, min_samples_leaf = 3), "RandomForestAudit", flat = True)
build_audit(RidgeClassifierCV(), "RidgeAudit", with_proba = False)
build_audit(BaggingClassifier(RidgeClassifier(random_state = 13), random_state = 13, n_estimators = 3, max_features = 0.5), "RidgeEnsembleAudit")
build_audit(SVC(), "SVCAudit", with_proba = False)
build_audit(VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression())], voting = "soft", weights = [3, 1, 2]), "VotingEnsembleAudit")
build_audit(OptimalXGBClassifier(objective = "binary:logistic", ntree_limit = 71), "XGBAudit")

audit_dict_X = audit_X.to_dict("records")

def build_audit_dict(classifier, name, with_proba = True):
	pipeline = PMMLPipeline([
		("dict-transformer", DictVectorizer()),
		("classifier", classifier)
	])
	pipeline.fit(audit_dict_X, audit_y)
	store_pkl(pipeline, name + ".pkl")
	adjusted = DataFrame(pipeline.predict(audit_dict_X), columns = ["Adjusted"])
	if(with_proba == True):
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_dict_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name + ".csv")

build_audit_dict(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeAuditDict")
build_audit_dict(LogisticRegression(), "LogisticRegressionAuditDict")

audit_na_X, audit_na_y = load_audit("AuditNA.csv")

def build_audit_na(classifier, name, with_proba = True):
	employment_mapping = {
		"Consultant" : "Private",
		"PSFederal" : "Public",
		"PSLocal" : "Public",
		"PSState" : "Public",
		"SelfEmp" : "Private",
		"Private" : "Private"
	}
	gender_mapping = {
		"Female" : 0,
		"Male" : 1
	}
	mapper = DataFrameMapper(
		[([column], [ContinuousDomain(missing_values = None), Imputer()]) for column in ["Age", "Income", "Hours"]] +
		[("Employment", [CategoricalDomain(missing_values = None), CategoricalImputer(), LookupTransformer(employment_mapping, "Other"), PMMLLabelBinarizer()])] +
		[([column], [CategoricalDomain(missing_values = None), CategoricalImputer(), PMMLLabelBinarizer()]) for column in ["Education", "Marital", "Occupation"]] +
		[("Gender", [CategoricalDomain(missing_values = None), CategoricalImputer(), LookupTransformer(gender_mapping, None)])]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_na_X, audit_na_y)
	store_pkl(pipeline, name + ".pkl")
	adjusted = DataFrame(pipeline.predict(audit_na_X), columns = ["Adjusted"])
	if(with_proba == True):
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_na_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name + ".csv")

build_audit_na(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeAuditNA")
build_audit_na(LogisticRegression(), "LogisticRegressionAuditNA")

versicolor_X, versicolor_y = load_versicolor("Versicolor.csv")

def build_versicolor(classifier, name, with_proba = True, **kwargs):
	mapper = DataFrameMapper([
		(versicolor_X.columns.values, [ContinuousDomain(), RobustScaler()])
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("transformer-pipeline", Pipeline([
			("polynomial", PolynomialFeatures(degree = 3)),
			("selector", SelectKBest(k = "all"))
		])),
		("classifier", classifier)
	])
	pipeline.fit(versicolor_X, versicolor_y)
	pipeline = make_pmml_pipeline(pipeline, versicolor_X.columns.values, versicolor_y.name)
	pipeline.verify(versicolor_X.sample(frac = 0.10, random_state = 13))
	customize(classifier, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	if(with_proba == True):
		species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name + ".csv")

build_versicolor(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeVersicolor", compact = False)
build_versicolor(DummyClassifier(strategy = "prior"), "DummyVersicolor")
build_versicolor(KNeighborsClassifier(), "KNNVersicolor", with_proba = False)
build_versicolor(MLPClassifier(activation = "tanh", hidden_layer_sizes = (8,), solver = "lbfgs", random_state = 13, tol = 0.1, max_iter = 100), "MLPVersicolor")
build_versicolor(SGDClassifier(random_state = 13, n_iter = 100), "SGDVersicolor", with_proba = False)
build_versicolor(SGDClassifier(random_state = 13, loss = "log", n_iter = 100), "SGDLogVersicolor")
build_versicolor(SVC(), "SVCVersicolor", with_proba = False)
build_versicolor(NuSVC(), "NuSVCVersicolor", with_proba = False)

#
# Multi-class classification
#

iris_X, iris_y = load_iris("Iris.csv")

def build_iris(classifier, name, with_proba = True, **kwargs):
	pipeline = Pipeline([
		("pipeline", Pipeline([
			("domain", ContinuousDomain()),
			("transform", FeatureUnion([
				("normal_scale", FunctionTransformer(None)),
				("log_scale", FunctionTransformer(numpy.log10))
			]))
		])),
		("scaler", RobustScaler()),
		("pca", IncrementalPCA(n_components = 3, whiten = True)),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	pipeline = make_pmml_pipeline(pipeline, iris_X.columns.values, iris_y.name)
	if isinstance(classifier, XGBClassifier):
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	customize(classifier, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	if(with_proba == True):
		species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name + ".csv")

build_iris(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeIris", compact = False)
build_iris(BaggingClassifier(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleIris")
build_iris(DummyClassifier(strategy = "constant", constant = "versicolor"), "DummyIris")
build_iris(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesIris")
build_iris(GradientBoostingClassifier(random_state = 13, init = None, n_estimators = 17), "GradientBoostingIris")
build_iris(KNeighborsClassifier(), "KNNIris", with_proba = False)
build_iris(OptimalLGBMClassifier(objective = "multiclass", n_estimators = 7, num_iteration = 3), "LGBMIris")
build_iris(LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysisIris")
build_iris(LogisticRegressionCV(), "LogisticRegressionIris")
build_iris(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleIris")
build_iris(MLPClassifier(hidden_layer_sizes = (6,), solver = "lbfgs", random_state = 13, tol = 0.1, max_iter = 100), "MLPIris")
build_iris(GaussianNB(), "NaiveBayesIris")
build_iris(RandomForestClassifier(random_state = 13, min_samples_leaf = 5), "RandomForestIris", flat = True)
build_iris(RidgeClassifierCV(), "RidgeIris", with_proba = False)
build_iris(BaggingClassifier(RidgeClassifier(random_state = 13), random_state = 13, n_estimators = 3, max_features = 0.5), "RidgeEnsembleIris")
build_iris(SGDClassifier(random_state = 13, n_iter = 100), "SGDIris", with_proba = False)
build_iris(SGDClassifier(random_state = 13, loss = "log", n_iter = 100), "SGDLogIris")
build_iris(SVC(), "SVCIris", with_proba = False)
build_iris(NuSVC(), "NuSVCIris", with_proba = False)
build_iris(VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression())]), "VotingEnsembleIris", with_proba = False)
build_iris(OptimalXGBClassifier(objective = "multi:softprob", ntree_limit = 7), "XGBIris")

#
# Text classification
#

sentiment_X, sentiment_y = load_sentiment("Sentiment.csv")

def build_sentiment(classifier, name, with_proba = True, **kwargs):
	pipeline = PMMLPipeline([
		("tf-idf", TfidfVectorizer(analyzer = "word", preprocessor = None, strip_accents = None, lowercase = True, token_pattern = None, tokenizer = Splitter(), stop_words = "english", ngram_range = (1, 2), norm = None, dtype = (numpy.float32 if isinstance(classifier, RandomForestClassifier) else numpy.float64))),
		("selector", SelectKBest(f_classif, k = 500)),
		("classifier", classifier)
	])
	pipeline.fit(sentiment_X, sentiment_y)
	customize(classifier, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	score = DataFrame(pipeline.predict(sentiment_X), columns = ["Score"])
	if(with_proba == True):
		score_proba = DataFrame(pipeline.predict_proba(sentiment_X), columns = ["probability(0)", "probability(1)"])
		score = pandas.concat((score, score_proba), axis = 1)
	store_csv(score, name + ".csv")

build_sentiment(LogisticRegressionCV(), "LogisticRegressionSentiment")
build_sentiment(RandomForestClassifier(random_state = 13, min_samples_leaf = 3), "RandomForestSentiment", compact = False)

#
# Regression
#

auto_X, auto_y = load_auto("Auto.csv")

def build_auto(regressor, name, **kwargs):
	cylinders_origin_mapping = {
		(8, 1) : "8/1",
		(6, 1) : "6/1",
		(4, 1) : "4/1",
		(6, 2) : "6/2",
		(4, 2) : "4/2",
		(6, 3) : "6/3",
		(4, 3) : "4/3"
	}
	mapper = DataFrameMapper([
		(["cylinders", "origin"], [MultiDomain([CategoricalDomain(), CategoricalDomain()]), MultiLookupTransformer(cylinders_origin_mapping, default_value = "other"), LabelBinarizer()]),
		(["model_year"], [CategoricalDomain(), Binarizer(threshold = 77)], {"alias" : "bin(model_year, 77)"}), # Pre/post 1973 oil crisis effects
		(["displacement", "horsepower", "weight", "acceleration"], [ContinuousDomain(), StandardScaler()]),
		(["weight", "displacement"], ExpressionTransformer("(X[:, 0] / X[:, 1]) + 0.5"), {"alias" : "weight / displacement + 0.5"})
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	pipeline = make_pmml_pipeline(pipeline, auto_X.columns.values, auto_y.name)
	if isinstance(regressor, XGBRegressor):
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	customize(regressor, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name + ".csv")

build_auto(AdaBoostRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 17), "AdaBoostAuto")
build_auto(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 2), "DecisionTreeAuto", compact = False)
build_auto(BaggingRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleAuto")
build_auto(DummyRegressor(strategy = "median"), "DummyAuto")
build_auto(ElasticNetCV(random_state = 13), "ElasticNetAuto")
build_auto(ExtraTreesRegressor(random_state = 13, min_samples_leaf = 5), "ExtraTreesAuto")
build_auto(GradientBoostingRegressor(random_state = 13, init = None), "GradientBoostingAuto")
build_auto(LassoCV(random_state = 13), "LassoAuto")
build_auto(OptimalLGBMRegressor(objective = "regression", n_estimators = 17, num_iteration = 11), "LGBMAuto")
build_auto(LinearRegression(), "LinearRegressionAuto")
build_auto(BaggingRegressor(LinearRegression(), random_state = 13, max_features = 0.75), "LinearRegressionEnsembleAuto")
build_auto(RandomForestRegressor(random_state = 13, min_samples_leaf = 3), "RandomForestAuto", flat = True)
build_auto(RidgeCV(), "RidgeAuto")
build_auto(OptimalXGBRegressor(objective = "reg:linear", ntree_limit = 31), "XGBAuto")

auto_na_X, auto_na_y = load_auto("AutoNA.csv")

auto_na_X["cylinders"] = auto_na_X["cylinders"].fillna(-1).astype(int)
auto_na_X["model_year"] = auto_na_X["model_year"].fillna(-1).astype(int)
auto_na_X["origin"] = auto_na_X["origin"].fillna(-1).astype(int)

def build_auto_na(regressor, name):
	mapper = DataFrameMapper(
		[([column], [CategoricalDomain(missing_values = -1), CategoricalImputer(missing_values = -1), PMMLLabelBinarizer()]) for column in ["cylinders", "model_year"]] +
		[(["origin"], [CategoricalImputer(missing_values = -1), OneHotEncoder()])] +
		[([column], [ContinuousDomain(missing_values = None), Imputer()]) for column in ["acceleration", "displacement", "horsepower", "weight"]]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_na_X, auto_na_y)
	store_pkl(pipeline, name + ".pkl")
	mpg = DataFrame(pipeline.predict(auto_na_X), columns = ["mpg"])
	store_csv(mpg, name + ".csv")

build_auto_na(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 2), "DecisionTreeAutoNA")
build_auto_na(LinearRegression(), "LinearRegressionAutoNA")

housing_X, housing_y = load_housing("Housing.csv")

def build_housing(regressor, name, with_kneighbors = False, **kwargs):
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
	pipeline.verify(housing_X.sample(frac = 0.05, random_state = 13))
	customize(regressor, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	medv = DataFrame(pipeline.predict(housing_X), columns = ["MEDV"])
	if(with_kneighbors == True):
		Xt = pipeline_transform(pipeline, housing_X)
		kneighbors = regressor.kneighbors(Xt)
		medv_ids = DataFrame(kneighbors[1] + 1, columns = ["neighbor(" + str(x + 1) + ")" for x in range(regressor.n_neighbors)])
		medv = pandas.concat((medv, medv_ids), axis = 1)
	store_csv(medv, name + ".csv")

build_housing(AdaBoostRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 17), "AdaBoostHousing")
build_housing(KNeighborsRegressor(), "KNNHousing", with_kneighbors = True)
build_housing(MLPRegressor(activation = "tanh", hidden_layer_sizes = (26,), solver = "lbfgs", random_state = 13, tol = 0.001, max_iter = 1000), "MLPHousing")
build_housing(SGDRegressor(random_state = 13), "SGDHousing")
build_housing(SVR(), "SVRHousing")
build_housing(LinearSVR(random_state = 13), "LinearSVRHousing")
build_housing(NuSVR(), "NuSVRHousing")

#
# Anomaly detection
#

def build_iforest_housing_anomaly(iforest, name, **kwargs):
	mapper = DataFrameMapper([
		(housing_X.columns.values, ContinuousDomain())
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("estimator", iforest)
	])
	pipeline.fit(housing_X)
	pipeline = make_pmml_pipeline(pipeline, housing_X.columns.values)
	customize(iforest, **kwargs)
	store_pkl(pipeline, name + ".pkl")
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) == -1, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name + ".csv")

build_iforest_housing_anomaly(IsolationForest(random_state = 13), "IsolationForestHousingAnomaly")

def build_svm_housing_anomaly(svm, name):
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
	store_pkl(pipeline, name + ".pkl")
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) <= 0, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name + ".csv")

build_svm_housing_anomaly(OneClassSVM(nu = 0.10, random_state = 13), "OneClassSVMHousingAnomaly")
