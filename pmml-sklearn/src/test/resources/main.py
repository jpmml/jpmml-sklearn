from common import *

from pandas import DataFrame
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import IncrementalPCA, PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingClassifier, BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, GammaRegressor, HuberRegressor, LarsCV, Lasso, LassoCV, LassoLarsCV, LinearRegression, LogisticRegression, LogisticRegressionCV, OrthogonalMatchingPursuitCV, PoissonRegressor, QuantileRegressor, Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier, SGDOneClassSVM, SGDRegressor, TheilSenRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier, MultiOutputRegressor, RegressorChain
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid, NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Binarizer, FunctionTransformer, KBinsDiscretizer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from sklearn2pmml import EstimatorProxy, SelectorProxy
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain, ContinuousDomainEraser, DiscreteDomainEraser, MultiAlias, MultiDomain
from sklearn2pmml.ensemble import EstimatorChain, GBDTLMRegressor, GBDTLRClassifier, Link, SelectFirstClassifier
from sklearn2pmml.feature_extraction.text import Matcher, Splitter
from sklearn2pmml.feature_selection import SelectUnique
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import Aggregator, CastTransformer, ConcatTransformer, CutTransformer, DataFrameConstructor, DaysSinceYearTransformer, ExpressionTransformer, FilterLookupTransformer, LookupTransformer, MatchesTransformer, MultiLookupTransformer, PMMLLabelBinarizer, PMMLLabelEncoder, PowerFunctionTransformer, ReplaceTransformer, SubstringTransformer, StringNormalizer, WordCountTransformer
from sklearn2pmml.ruleset import RuleSetClassifier
from sklearn2pmml.util import Slicer
from sklearn_pandas import CategoricalImputer, DataFrameMapper
from xgboost.sklearn import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

import numpy
import pandas
import sys

def pipeline_transform(pipeline, X):
	identity_pipeline = Pipeline(pipeline.steps[: -1] + [("estimator", None)])
	return identity_pipeline.transform(X)

def make_interaction(left, right):
	pipeline = Pipeline([
		("mapper", DataFrameMapper([
			([left], LabelBinarizer()),
			([right], LabelBinarizer())
		])),
		("polynomial", PolynomialFeatures())
	])
	return pipeline

def make_kneighbor_cols(estimator):
	return ["neighbor(" + str(x + 1) + ")" for x in range(estimator.n_neighbors)]

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Housing", "Iris", "Sentiment", "Versicolor", "Visit", "Wheat"]

#
# Clustering
#

def kmeans_affinity(kmeans, X):
	affinity_0 = kmeans_distance(kmeans, 0, X)
	affinity_1 = kmeans_distance(kmeans, 1, X)
	affinity_2 = kmeans_distance(kmeans, 2, X)
	return DataFrame(numpy.transpose([affinity_0, affinity_1, affinity_2]), columns = ["affinity(0)", "affinity(1)", "affinity(2)"])

def kmeans_distance(kmeans, center, X):
	return numpy.sum(numpy.power(kmeans.cluster_centers_[center] - X, 2), axis = 1)

def build_wheat(wheat_df, clusterer, name, with_affinity = False, with_kneighbors = False, **pmml_options):
	wheat_X, wheat_y = split_csv(wheat_df)

	mapper = DataFrameMapper([
		(wheat_X.columns.values, [ContinuousDomain(dtype = float), PowerTransformer(method = "box-cox", standardize = True)])
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("clusterer", clusterer)
	])
	pipeline.fit(wheat_X)
	pipeline = make_pmml_pipeline(pipeline, wheat_X.columns.values)
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)
	if hasattr(clusterer, "predict"):
		cluster = DataFrame(pipeline.predict(wheat_X), columns = ["cluster"])
	if with_affinity:
		Xt = pipeline_transform(pipeline, wheat_X)
		cluster_affinities = kmeans_affinity(clusterer, Xt)
		cluster = pandas.concat((cluster, cluster_affinities), axis = 1)
	if with_kneighbors:
		Xt = pipeline_transform(pipeline, wheat_X)
		kneighbors = clusterer.kneighbors(Xt)
		cluster_ids = DataFrame(kneighbors[1], columns = make_kneighbor_cols(clusterer))
		cluster = cluster_ids
	store_csv(cluster, name)

if "Wheat" in datasets:
	wheat_df = load_wheat("Wheat")

	build_wheat(wheat_df, KMeans(n_clusters = 3, random_state = 13), "KMeansWheat", with_affinity = True)
	build_wheat(wheat_df, MiniBatchKMeans(n_clusters = 3, compute_labels = False, random_state = 13), "MiniBatchKMeansWheat", with_affinity = True)
	build_wheat(wheat_df, NearestNeighbors(n_neighbors = 3, algorithm = "ball_tree"), "NearestNeighborsWheat", with_kneighbors = True)

#
# Binary classification
#

def build_audit(audit_df, classifier, name, with_proba = True, fit_params = {}, predict_params = {}, predict_proba_params = {}, **pmml_options):
	audit_X, audit_y = split_csv(audit_df)

	continuous_mapper = DataFrameMapper([
		(["Age", "Hours"], MultiDomain([ContinuousDomain() for i in range(0, 2)])),
		(["Income"], [ContinuousDomain(), KBinsDiscretizer(n_bins = 3, strategy = "quantile")])
	])
	categorical_mapper = DataFrameMapper([
		(["Employment"], [CategoricalDomain(), SubstringTransformer(0, 3), OneHotEncoder(drop = ["Vol"]), SelectFromModel(DecisionTreeClassifier(random_state = 13))]),
		(["Education"], [CategoricalDomain(), ReplaceTransformer("[aeiou]", ""), OneHotEncoder(drop = "first"), SelectFromModel(RandomForestClassifier(n_estimators = 3, random_state = 13), threshold = "1.25 * mean")]),
		(["Marital"], [CategoricalDomain(), LabelBinarizer(neg_label = -1, pos_label = 1), SelectKBest(k = 3)]),
		(["Occupation"], [CategoricalDomain(), LabelBinarizer(), SelectKBest(k = 3)]),
		(["Gender"], [CategoricalDomain(), MatchesTransformer("^Male$"), CastTransformer(int)]),
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
	pipeline.fit(audit_X, audit_y, **fit_params)
	pipeline = make_pmml_pipeline(pipeline, audit_X.columns.values, audit_y.name)
	pipeline.configure(**pmml_options)
	if isinstance(classifier, EstimatorProxy):
		estimator = classifier.estimator
		if hasattr(estimator, "feature_importances_"):
			estimator.pmml_feature_importances_ = estimator.feature_importances_
		if hasattr(estimator, "estimators_"):
			child_estimators = estimator.estimators_
			if isinstance(child_estimators, numpy.ndarray):
				child_estimators = child_estimators.flatten().tolist()
			for child_estimator in child_estimators:
				if hasattr(child_estimator, "feature_importances_"):
					child_estimator.pmml_feature_importances_ = child_estimator.feature_importances_
	elif isinstance(classifier, XGBClassifier):
		classifier.pmml_feature_importances_ = classifier.feature_importances_
	else:
		pass
	if isinstance(classifier, GaussianNB):
		pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13), predict_params = predict_params, predict_proba_params = predict_proba_params, precision = 1e-12, zeroThreshold = 1e-12)
	elif isinstance(classifier, XGBClassifier):
		pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13), predict_params = predict_params, predict_proba_params = predict_proba_params, precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13), predict_params = predict_params, predict_proba_params = predict_proba_params)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X, **predict_params), columns = ["Adjusted"])
	if with_proba:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_X, **predict_proba_params), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit", stringify = False)

	build_audit(audit_df, EstimatorProxy(DecisionTreeClassifier(min_samples_leaf = 2, random_state = 13)), "DecisionTreeAudit", compact = False)
	build_audit(audit_df, BaggingClassifier(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), n_estimators = 3, max_features = 0.5, random_state = 13), "DecisionTreeEnsembleAudit")
	build_audit(audit_df, DummyClassifier(strategy = "most_frequent"), "DummyAudit")
	build_audit(audit_df, EstimatorProxy(ExtraTreesClassifier(n_estimators = 10, min_samples_leaf = 5, random_state = 13)), "ExtraTreesAudit")
	build_audit(audit_df, GBDTLRClassifier(RandomForestClassifier(n_estimators = 17, random_state = 13), LogisticRegression()), "GBDTLRAudit")
	build_audit(audit_df, EstimatorProxy(GradientBoostingClassifier(loss = "exponential", init = None, random_state = 13)), "GradientBoostingAudit")
	build_audit(audit_df, HistGradientBoostingClassifier(max_iter = 71, random_state = 13), "HistGradientBoostingAudit")
	build_audit(audit_df, LinearDiscriminantAnalysis(solver = "lsqr"), "LinearDiscriminantAnalysisAudit")
	build_audit(audit_df, LinearSVC(penalty = "l1", dual = False, random_state = 13), "LinearSVCAudit", with_proba = False)
	build_audit(audit_df, LogisticRegression(multi_class = "multinomial", solver = "newton-cg", max_iter = 500), "MultinomialLogisticRegressionAudit")
	build_audit(audit_df, LogisticRegressionCV(cv = 3, multi_class = "ovr"), "OvRLogisticRegressionAudit")
	build_audit(audit_df, BaggingClassifier(LogisticRegression(), n_estimators = 3, max_features = 0.5, random_state = 13), "LogisticRegressionEnsembleAudit")
	build_audit(audit_df, GaussianNB(), "NaiveBayesAudit")
	build_audit(audit_df, OneVsRestClassifier(LogisticRegression()), "OneVsRestAudit")
	build_audit(audit_df, EstimatorProxy(RandomForestClassifier(n_estimators = 10, min_samples_leaf = 3, random_state = 13)), "RandomForestAudit", flat = True)
	build_audit(audit_df, RidgeClassifierCV(), "RidgeAudit", with_proba = False)
	build_audit(audit_df, BaggingClassifier(RidgeClassifier(random_state = 13), n_estimators = 3, max_features = 0.5, random_state = 13), "RidgeEnsembleAudit")
	build_audit(audit_df, StackingClassifier([("lda", LinearDiscriminantAnalysis(solver = "lsqr")), ("lr", LogisticRegression())], final_estimator = GradientBoostingClassifier(n_estimators = 11, random_state = 13)), "StackingEnsembleAudit")
	build_audit(audit_df, SVC(gamma = "auto"), "SVCAudit", with_proba = False)
	build_audit(audit_df, VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression())], voting = "soft", weights = [3, 1, 2]), "VotingEnsembleAudit")

def build_audit_dict(audit_df, classifier, name, with_proba = True):
	audit_X, audit_y = split_csv(audit_df)
	audit_dict_X = audit_X.to_dict("records")

	header = {
		"copyright" : "Copyright (c) 2021 Villu Ruusmann",
		"description" : "Integration test for dictionary (key-value mappings) input",
		"modelVersion" : "1.0.0"
	}
	pipeline = PMMLPipeline([
		("dict-transformer", DictVectorizer()),
		("classifier", classifier)
	], header = header)
	pipeline.fit(audit_dict_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_dict_X), columns = ["Adjusted"])
	if with_proba:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_dict_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit", stringify = False)
	
	build_audit_dict(audit_df, DecisionTreeClassifier(min_samples_leaf = 5, random_state = 13), "DecisionTreeAuditDict")
	build_audit_dict(audit_df, LogisticRegression(), "LogisticRegressionAuditDict")
	build_audit_dict(audit_df, NearestCentroid(), "NearestCentroidAuditDict", with_proba = False)

def build_audit_na(audit_na_df, classifier, name, with_proba = True, fit_params = {}, predict_params = {}, predict_proba_params = {}, predict_transformer = None, predict_proba_transformer = None, apply_transformer = None, **pmml_options):
	audit_na_X, audit_na_y = split_csv(audit_na_df)

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
		[(["Age"], [ContinuousDomain(missing_values = None, with_data = False), Alias(ExpressionTransformer("X[0] if pandas.notnull(X[0]) else -999", dtype = int), name = "flag_missing(Age, -999)"), SimpleImputer(missing_values = -999, strategy = "constant", fill_value = 38)])] +
		[(["Age"], MissingIndicator())] +
		[(["Hours"], [ContinuousDomain(missing_values = None, with_data = False), Alias(ExpressionTransformer("-999 if pandas.isnull(X[0]) else X[0]"), name = "flag_missing(Hours, -999)"), SimpleImputer(missing_values = -999, add_indicator = True)])] +
		[(["Income"], [ContinuousDomain(missing_values = None, outlier_treatment = "as_missing_values", low_value = 5000, high_value = 200000, with_data = False), SimpleImputer(strategy = "median", add_indicator = True)])] +
		[(["Employment"], [CategoricalDomain(missing_values = None, with_data = False), CategoricalImputer(missing_values = None), StringNormalizer(function = "uppercase"), LookupTransformer(employment_mapping, "OTHER"), StringNormalizer(function = "lowercase"), PMMLLabelBinarizer(), DiscreteDomainEraser()])] +
		[([column], [CategoricalDomain(missing_values = None, missing_value_replacement = "N/A", with_data = False), SimpleImputer(missing_values = "N/A", strategy = "most_frequent"), StringNormalizer(function = "lowercase"), PMMLLabelBinarizer(), DiscreteDomainEraser()]) for column in ["Education", "Marital", "Occupation"]] +
		[(["Gender"], [CategoricalDomain(missing_values = None, with_data = False), SimpleImputer(strategy = "constant"), StringNormalizer(function = "uppercase"), LookupTransformer(gender_mapping, None)])]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	], predict_transformer = predict_transformer, predict_proba_transformer = predict_proba_transformer, apply_transformer = apply_transformer)
	pipeline.fit(audit_na_X, audit_na_y, **fit_params)
	pipeline.configure(**pmml_options)
	pipeline.verify(audit_na_X.sample(frac = 0.05, random_state = 13), predict_params = predict_params, predict_proba_params = predict_proba_params)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_na_X, **predict_params), columns = ["Adjusted"])
	if with_proba:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_na_X, **predict_proba_params), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	if isinstance(classifier, DecisionTreeClassifier):
		Xt = pipeline_transform(pipeline, audit_na_X)
		adjusted_apply = DataFrame(classifier.apply(Xt), columns = ["nodeId"])
		adjusted = pandas.concat((adjusted, adjusted_apply), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_na_df = load_audit("AuditNA")

	build_audit_na(audit_na_df, DecisionTreeClassifier(min_samples_leaf = 5, random_state = 13), "DecisionTreeAuditNA", apply_transformer = Alias(ExpressionTransformer("X[0] - 1"), "eval(nodeId)", prefit = True), winner_id = True, class_extensions = {"event" : {"0" : False, "1" : True}})
	build_audit_na(audit_na_df, LogisticRegression(solver = "newton-cg", max_iter = 500), "LogisticRegressionAuditNA", predict_proba_transformer = Alias(ExpressionTransformer("1 if X[1] > 0.75 else 0"), name = "eval(probability(1))", prefit = True))

def build_audit_na_hist(audit_na_df, classifier, name):
	audit_na_X, audit_na_y = split_csv(audit_na_df)

	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["Age", "Hours", "Income"]] +
		[([column], [CategoricalDomain(), PMMLLabelEncoder()]) for column in ["Employment", "Education", "Marital", "Occupation", "Gender"]]
	)
	pipeline = PMMLPipeline([
		("pipeline", Pipeline([
			("mapper", mapper),
			("classifier", classifier)
		]))
	])
	pipeline.fit(audit_na_X, audit_na_y)
	pipeline.verify(audit_na_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_na_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_na_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	build_audit_na_hist(audit_na_df, HistGradientBoostingClassifier(max_iter = 71, categorical_features = [3, 4, 5, 6, 7], random_state = 13), "HistGradientBoostingAuditNA")

def build_multi_audit(audit_df, classifier, name, with_kneighbors = False):
	audit_X, audit_y = split_multi_csv(audit_df, ["Gender", "Adjusted"])

	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["Age", "Hours", "Income"]] +
		[([column], [CategoricalDomain(), LabelBinarizer()]) for column in ["Employment", "Education", "Marital", "Occupation"]]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	gender_adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Gender", "Adjusted"])
	if with_kneighbors:
		Xt = pipeline_transform(pipeline, audit_X)
		kneighbors = classifier.kneighbors(Xt)
		gender_adjusted_ids = DataFrame(kneighbors[1] + 1, columns = make_kneighbor_cols(classifier))
		gender_adjusted = pandas.concat((gender_adjusted, gender_adjusted_ids), axis = 1)
	store_csv(gender_adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")
	audit_df["Gender"] = audit_df["Gender"].astype(str)
	audit_df["Adjusted"] = audit_df["Adjusted"].astype(str)

	build_multi_audit(audit_df, EstimatorChain([("gender", Link(DecisionTreeClassifier(max_depth = 5, random_state = 13), augment_funcs = ["predict_proba", "apply"]), str(True)), ("adjusted", LogisticRegression(), str(True))]), "MultiEstimatorChainAudit")
	build_multi_audit(audit_df, KNeighborsClassifier(metric = "euclidean"), "MultiKNNAudit", with_kneighbors = True)
	build_multi_audit(audit_df, MultiOutputClassifier(LogisticRegression()), "MultiLogisticRegressionAudit")

	# Translate labels from string to numeric
	# Use unique labels (-2/2 vs 0/1) between individual classifiers
	audit_df["Gender"] = audit_df["Gender"].replace("Male", -2).replace("Female", 2).astype(int)
	audit_df["Adjusted"] = audit_df["Adjusted"].astype(int)

	build_multi_audit(audit_df, ClassifierChain(LogisticRegression()), "LogisticRegressionChainAudit")

def build_versicolor(versicolor_df, classifier, name, with_proba = True, **pmml_options):
	versicolor_X, versicolor_y = split_csv(versicolor_df)

	scaler = ColumnTransformer([
		("robust", RobustScaler(), [0, 2])
	], remainder = MinMaxScaler())

	transformer = ColumnTransformer([
		("continuous_columns", Pipeline([
			("domain", ContinuousDomain()),
			("scaler", MultiAlias(scaler, names = ["scaler(" + col + ")" for col in versicolor_X.columns.values]))
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
	if with_proba:
		species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Versicolor" in datasets:
	versicolor_df = load_versicolor("Versicolor")

	build_versicolor(versicolor_df, DummyClassifier(strategy = "prior"), "DummyVersicolor")
	build_versicolor(versicolor_df, GBDTLRClassifier(GradientBoostingClassifier(n_estimators = 11, random_state = 13), LogisticRegression(solver = "liblinear")), "GBDTLRVersicolor")
	build_versicolor(versicolor_df, KNeighborsClassifier(metric = "euclidean"), "KNNVersicolor", with_proba = False)
	build_versicolor(versicolor_df, MLPClassifier(activation = "tanh", hidden_layer_sizes = (8,), solver = "lbfgs", tol = 0.1, max_iter = 100, random_state = 13), "MLPVersicolor")
	build_versicolor(versicolor_df, SGDClassifier(max_iter = 100, random_state = 13), "SGDVersicolor", with_proba = False)
	build_versicolor(versicolor_df, SGDClassifier(loss = "log_loss", max_iter = 100, random_state = 13), "SGDLogVersicolor")
	build_versicolor(versicolor_df, GridSearchCV(SVC(gamma = "auto"), {"C" : [1, 3, 5]}), "SVCVersicolor", with_proba = False)
	build_versicolor(versicolor_df, RandomizedSearchCV(NuSVC(gamma = "auto"), {"nu" : [0.3, 0.4, 0.5, 0.6]}), "NuSVCVersicolor", with_proba = False)

def build_versicolor_direct(versicolor_df, classifier, name, with_proba = True, **pmml_options):
	versicolor_X, versicolor_y = split_csv(versicolor_df)

	transformer = ColumnTransformer([
		("all", "passthrough", ["Sepal.Length", "Petal.Length", "Petal.Width"])
	], remainder = "drop")
	pipeline = PMMLPipeline([
		("transformer", transformer),
		("passthrough-transformer", Slicer(start = None, stop = None)),
		("svd", TruncatedSVD(n_components = 2)),
		("classifier", classifier)
	])
	pipeline.fit(versicolor_X, versicolor_y)
	pipeline.configure(**pmml_options)
	pipeline.verify(versicolor_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	if with_proba:
		species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Versicolor" in datasets:
	build_versicolor_direct(versicolor_df, DecisionTreeClassifier(min_samples_leaf = 5, random_state = 13), "DecisionTreeVersicolor", compact = False)

#
# Multi-class classification
#

def build_iris(iris_df, classifier, name, with_proba = True, fit_params = {}, predict_params = {}, predict_proba_params = {}, **pmml_options):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = Pipeline([
		("pipeline", Pipeline([
			("mapper", DataFrameMapper([
				(iris_X.columns.values, ContinuousDomain()),
				(["Sepal.Length", "Petal.Length"], Aggregator(function = "mean")),
				(["Sepal.Width", "Petal.Width"], Aggregator(function = "mean"))
			])),
			("transform", FeatureUnion([
				("normal_scale", FunctionTransformer(None, validate = True)),
				("log_scale", FunctionTransformer(numpy.log10, validate = True)),
				("power_scale", PowerFunctionTransformer(power = 2))
			]))
		])),
		("pca", IncrementalPCA(n_components = 3, whiten = True)),
		("renamer", DataFrameConstructor(columns = ["pca(1)", "pca(2)", "pca(3)"], dtype = numpy.float64)),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y, **fit_params)
	pipeline = make_pmml_pipeline(pipeline, iris_X.columns.values, iris_y.name)
	pipeline.configure(**pmml_options)
	if isinstance(classifier, XGBClassifier):
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13), predict_params = predict_params, predict_proba_params = predict_proba_params, precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13), predict_params = predict_params, predict_proba_params = predict_proba_params)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X, **predict_params), columns = ["Species"])
	if with_proba:
		species_proba = DataFrame(pipeline.predict_proba(iris_X, **predict_proba_params), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_iris(iris_df, DecisionTreeClassifier(min_samples_leaf = 5, random_state = 13), "DecisionTreeIris", compact = False)
	build_iris(iris_df, BaggingClassifier(DecisionTreeClassifier(min_samples_leaf = 5, random_state = 13), n_estimators = 3, max_features = 0.5, random_state = 13), "DecisionTreeEnsembleIris")
	build_iris(iris_df, DummyClassifier(strategy = "constant", constant = "versicolor"), "DummyIris")
	build_iris(iris_df, ExtraTreesClassifier(n_estimators = 10, min_samples_leaf = 5, random_state = 13), "ExtraTreesIris")
	build_iris(iris_df, GradientBoostingClassifier(init = None, n_estimators = 17, random_state = 13), "GradientBoostingIris")
	build_iris(iris_df, HistGradientBoostingClassifier(max_iter = 10, random_state = 13), "HistGradientBoostingIris")
	build_iris(iris_df, KNeighborsClassifier(metric = "manhattan"), "KNNIris", with_proba = False)
	build_iris(iris_df, LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysisIris")
	build_iris(iris_df, LinearSVC(random_state = 13), "LinearSVCIris", with_proba = False)
	build_iris(iris_df, LogisticRegression(multi_class = "multinomial", solver = "lbfgs"), "MultinomialLogisticRegressionIris")
	build_iris(iris_df, LogisticRegressionCV(cv = 3, multi_class = "ovr"), "OvRLogisticRegressionIris")
	build_iris(iris_df, BaggingClassifier(LogisticRegression(multi_class = "ovr", solver = "liblinear"), n_estimators = 3, max_features = 0.5, random_state = 13), "LogisticRegressionEnsembleIris")
	build_iris(iris_df, MLPClassifier(hidden_layer_sizes = (6,), solver = "lbfgs", tol = 0.1, max_iter = 100, random_state = 13), "MLPIris")
	build_iris(iris_df, GaussianNB(), "NaiveBayesIris")
	build_iris(iris_df, OneVsRestClassifier(LogisticRegression(multi_class = "ovr", solver = "liblinear")), "OneVsRestIris")
	build_iris(iris_df, RandomForestClassifier(n_estimators = 10, min_samples_leaf = 5, random_state = 13), "RandomForestIris", flat = True)
	build_iris(iris_df, RidgeClassifierCV(), "RidgeIris", with_proba = False)
	build_iris(iris_df, BaggingClassifier(RidgeClassifier(random_state = 13), n_estimators = 3, max_features = 0.5, random_state = 13), "RidgeEnsembleIris")
	build_iris(iris_df, SGDClassifier(max_iter = 100, random_state = 13), "SGDIris", with_proba = False)
	build_iris(iris_df, SGDClassifier(loss = "log_loss", max_iter = 100, random_state = 13), "SGDLogIris")
	build_iris(iris_df, StackingClassifier([("lda", LinearDiscriminantAnalysis()), ("lr", LogisticRegression(multi_class = "multinomial", solver = "lbfgs"))], final_estimator = GradientBoostingClassifier(n_estimators = 5, random_state = 13), passthrough = True), "StackingEnsembleIris")
	build_iris(iris_df, SVC(gamma = "auto"), "SVCIris", with_proba = False)
	build_iris(iris_df, NuSVC(gamma = "auto"), "NuSVCIris", with_proba = False)
	build_iris(iris_df, VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression(multi_class = "ovr", solver = "liblinear"))]), "VotingEnsembleIris", with_proba = False)

# XXX
iris_train_mask = numpy.random.choice([False, True], size = (150,), p = [0.5, 0.5])
iris_test_mask = ~iris_train_mask

def build_iris_opt(iris_df, classifier, name, fit_params = {}, **pmml_options):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = PMMLPipeline([
		("classifier", classifier)
	])
	pipeline.fit(iris_X[iris_train_mask], iris_y[iris_train_mask], **fit_params)
	if isinstance(classifier, XGBClassifier):
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
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
	store_pkl(pipeline, "SelectFirstIris")
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, "SelectFirstIris")

if "Iris" in datasets:
	iris_X, iris_y = split_csv(iris_df)

	classifier = RuleSetClassifier([
		("X['Petal.Length'] >= 2.45 and X['Petal.Width'] < 1.75", "versicolor"),
		("X['Petal.Length'] >= 2.45", "virginica")
	], default_score = "setosa")
	pipeline = PMMLPipeline([
		("domain", ContinuousDomain(display_name = ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"])),
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

def build_sentiment(sentiment_df, classifier, tokenizer, name, with_proba = True, **pmml_options):
	sentiment_X = sentiment_df["Sentence"]
	sentiment_y = sentiment_df["Score"]

	pipeline = PMMLPipeline([
		("union", FeatureUnion([
			("tf-idf", TfidfVectorizer(analyzer = "word", preprocessor = None, strip_accents = None, lowercase = True, tokenizer = tokenizer, stop_words = "english", ngram_range = (1, 2), norm = None, sublinear_tf = isinstance(classifier, LogisticRegressionCV), dtype = (numpy.float32 if isinstance(classifier, RandomForestClassifier) else numpy.float64))),
			("count", WordCountTransformer())
		])),
		("selector", SelectKBest(f_classif, k = 1000)),
		("classifier", classifier)
	])
	pipeline.fit(sentiment_X, sentiment_y)
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)
	score = DataFrame(pipeline.predict(sentiment_X), columns = ["Score"])
	if with_proba:
		score_proba = DataFrame(pipeline.predict_proba(sentiment_X), columns = ["probability(0)", "probability(1)"])
		score = pandas.concat((score, score_proba), axis = 1)
	store_csv(score, name)

if "Sentiment" in datasets:
	sentiment_df = load_sentiment("Sentiment")

	build_sentiment(sentiment_df, LinearSVC(random_state = 13), Splitter(), "LinearSVCSentiment", with_proba = False)
	build_sentiment(sentiment_df, LogisticRegressionCV(cv = 3), None, "LogisticRegressionSentiment")
	build_sentiment(sentiment_df, RandomForestClassifier(n_estimators = 10, min_samples_leaf = 3, random_state = 13), Matcher(), "RandomForestSentiment", compact = False)

#
# Regression
#

def build_auto(auto_df, regressor, name, fit_params = {}, predict_params = {}, **pmml_options):
	auto_X, auto_y = split_csv(auto_df)

	cylinders_origin_mapping = {
		(8, 1) : "8/1",
		(6, 1) : "6/1",
		(4, 1) : "4/1",
		(6, 2) : "6/2",
		(4, 2) : "4/2",
		(4, 3) : "4/3"
	}
	mapper = DataFrameMapper([
		(["cylinders"], [CategoricalDomain(), Alias(ExpressionTransformer("X[0] % 2.0 > 0.0", dtype = numpy.int8), name = "odd(cylinders)", prefit = True)]),
		(["cylinders", "origin"], [MultiDomain([None, CategoricalDomain()]), MultiLookupTransformer(cylinders_origin_mapping, default_value = "other"), OneHotEncoder()]),
		(["model_year"], [CategoricalDomain(), CastTransformer(str), ExpressionTransformer("'19' + X[0] + '-01-01'"), CastTransformer("datetime64[D]"), DaysSinceYearTransformer(1977), Binarizer(threshold = 0)], {"alias" : "bin(model_year, 1977)"}),
		(["model_year", "origin"], [ConcatTransformer("/"), OneHotEncoder(sparse_output = False), SelectorProxy(SelectFromModel(RandomForestRegressor(n_estimators = 3, random_state = 13), threshold = "1.25 * mean"))]),
		(["weight", "displacement"], [ContinuousDomain(), ExpressionTransformer("(X[0] / X[1]) + 0.5", dtype = numpy.float64)], {"alias" : "weight / displacement + 0.5"}),
		(["displacement", "horsepower", "weight", "acceleration"], [MultiDomain([None, ContinuousDomain(), None, ContinuousDomain()]), StandardScaler()])
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("selector", SelectUnique()),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y, **fit_params)
	pipeline.configure(**pmml_options)
	if isinstance(regressor, XGBRegressor):
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13), predict_params = predict_params, precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13), predict_params = predict_params)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X, **predict_params), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	auto_df["cylinders"] = auto_df["cylinders"].astype(int)
	auto_df["model_year"] = auto_df["model_year"].astype(int)
	auto_df["origin"] = auto_df["origin"].astype(int)

	build_auto(auto_df, AdaBoostRegressor(DecisionTreeRegressor(min_samples_leaf = 5, random_state = 13), random_state = 13, n_estimators = 17), "AdaBoostAuto")
	build_auto(auto_df, ARDRegression(), "BayesianARDAuto")
	build_auto(auto_df, BayesianRidge(), "BayesianRidgeAuto")
	build_auto(auto_df, DecisionTreeRegressor(min_samples_leaf = 2, random_state = 13), "DecisionTreeAuto", compact = False)
	build_auto(auto_df, BaggingRegressor(DecisionTreeRegressor(min_samples_leaf = 5, random_state = 13), n_estimators = 3, max_features = 0.5, random_state = 13), "DecisionTreeEnsembleAuto")
	build_auto(auto_df, DummyRegressor(strategy = "median"), "DummyAuto")
	build_auto(auto_df, ElasticNetCV(cv = 3, random_state = 13), "ElasticNetAuto")
	build_auto(auto_df, ExtraTreesRegressor(n_estimators = 10, min_samples_leaf = 5, random_state = 13), "ExtraTreesAuto")
	build_auto(auto_df, GBDTLMRegressor(RandomForestRegressor(n_estimators = 7, max_depth = 6, random_state = 13), LinearRegression()), "GBDTLMAuto")
	build_auto(auto_df, GradientBoostingRegressor(init = None, random_state = 13), "GradientBoostingAuto")
	build_auto(auto_df, HuberRegressor(), "HuberAuto")
	build_auto(auto_df, LarsCV(cv = 3), "LarsAuto")
	build_auto(auto_df, LassoCV(cv = 3, random_state = 13), "LassoAuto")
	build_auto(auto_df, LassoLarsCV(cv = 3), "LassoLarsAuto")
	build_auto(auto_df, LinearRegression(), "LinearRegressionAuto")
	build_auto(auto_df, BaggingRegressor(LinearRegression(), max_features = 0.75, random_state = 13), "LinearRegressionEnsembleAuto")
	build_auto(auto_df, OrthogonalMatchingPursuitCV(cv = 3), "OMPAuto")
	build_auto(auto_df, QuantileRegressor(quantile = 0.65), "CQRAuto")
	build_auto(auto_df, RandomForestRegressor(n_estimators = 10, min_samples_leaf = 3, random_state = 13), "RandomForestAuto", flat = True)
	build_auto(auto_df, RidgeCV(), "RidgeAuto")
	build_auto(auto_df, StackingRegressor([("ridge", Ridge(random_state = 13)), ("lasso", Lasso(random_state = 13))], final_estimator = GradientBoostingRegressor(n_estimators = 7, random_state = 13)), "StackingEnsembleAuto")
	build_auto(auto_df, TheilSenRegressor(n_subsamples = 31, random_state = 13), "TheilSenAuto")
	build_auto(auto_df, VotingRegressor([("dt", DecisionTreeRegressor(random_state = 13)), ("knn", KNeighborsRegressor(algorithm = "kd_tree")), ("lr", LinearRegression())], weights = [3, 1, 2]), "VotingEnsembleAuto")

if "Auto" in datasets:
	build_auto(auto_df, TransformedTargetRegressor(DecisionTreeRegressor(random_state = 13)), "TransformedDecisionTreeAuto")
	build_auto(auto_df, TransformedTargetRegressor(LinearRegression(), func = numpy.log, inverse_func = numpy.exp), "TransformedLinearRegressionAuto")

def build_auto_hist(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["displacement", "horsepower", "weight", "acceleration"]] +
		[([column], [CategoricalDomain(), OrdinalEncoder()]) for column in ["cylinders", "model_year", "origin"]]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	build_auto_hist(auto_df, HistGradientBoostingRegressor(max_iter = 31, categorical_features = [4, 5, 6], random_state = 13), "HistGradientBoostingAuto")

def build_auto_isotonic(auto_isotonic_X, auto_y, regressor, name):
	pipeline = PMMLPipeline([
		("regressor", regressor)
	])
	pipeline.fit(auto_isotonic_X, auto_y)
	pipeline.verify(auto_isotonic_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_isotonic_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	build_auto_isotonic(auto_df["acceleration"], auto_df["mpg"], IsotonicRegression(increasing = True, out_of_bounds = "nan"), "IsotonicRegressionIncrAuto")
	build_auto_isotonic(auto_df["weight"], auto_df["mpg"], IsotonicRegression(increasing = False, y_min = 12, y_max = 36, out_of_bounds = "clip"), "IsotonicRegressionDecrAuto")

# XXX
auto_train_mask = numpy.random.choice([False, True], size = (392,), p = [0.8, 0.2])
auto_test_mask = ~auto_train_mask

def build_auto_opt(auto_df, regressor, name, fit_params = {}, **pmml_options):
	auto_X, auto_y = split_csv(auto_df)

	pipeline = PMMLPipeline([
		("regressor", regressor)
	])
	pipeline.fit(auto_X[auto_train_mask], auto_y[auto_train_mask], **fit_params)
	if isinstance(regressor, XGBRegressor):
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	else:
		pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

def build_auto_na(auto_na_df, regressor, name, predict_transformer = None, apply_transformer = None, **pmml_options):
	auto_na_X, auto_na_y = split_csv(auto_na_df)

	mapper = DataFrameMapper(
		[([column], [CategoricalDomain(missing_values = -1), CategoricalImputer(missing_values = -1), PMMLLabelBinarizer()]) for column in ["cylinders", "model_year"]] +
		[(["origin"], [CategoricalDomain(missing_values = -1), SimpleImputer(missing_values = -1, strategy = "most_frequent"), OneHotEncoder()])] +
		[(["acceleration"], [ContinuousDomain(missing_values = None), CutTransformer(bins = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25], labels = False), CategoricalImputer(), LabelBinarizer()])] +
		[(["displacement"], [ContinuousDomain(missing_values = None), SimpleImputer(), CutTransformer(bins = [0, 100, 200, 300, 400, 500], labels = ["XS", "S", "M", "L", "XL"]), LabelBinarizer()])] +
		[(["horsepower"], [ContinuousDomain(missing_values = None, outlier_treatment = "as_extreme_values", low_value = 50, high_value = 225), SimpleImputer(strategy = "median")])] +
		[(["weight"], [ContinuousDomain(missing_values = None, outlier_treatment = "as_extreme_values", low_value = 2000, high_value = 5000), SimpleImputer(strategy = "median")])]
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
	pipeline.verify(auto_na_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_na_X), columns = ["mpg"])
	if isinstance(regressor, DecisionTreeRegressor):
		Xt = pipeline_transform(pipeline, auto_na_X)
		mpg_apply = DataFrame(regressor.apply(Xt), columns = ["nodeId"])
		mpg = pandas.concat((mpg, mpg_apply), axis = 1)
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_na_df = load_auto("AutoNA")

	auto_na_df["cylinders"] = auto_na_df["cylinders"].fillna(-1).astype(int)
	auto_na_df["model_year"] = auto_na_df["model_year"].fillna(-1).astype(int)
	auto_na_df["origin"] = auto_na_df["origin"].fillna(-1).astype(int)

	build_auto_na(auto_na_df, DecisionTreeRegressor(min_samples_leaf = 2, random_state = 13), "DecisionTreeAutoNA", apply_transformer = Alias(ExpressionTransformer("X[0] - 1"), "eval(nodeId)", prefit = True), winner_id = True)
	build_auto_na(auto_na_df, LinearRegression(), "LinearRegressionAutoNA", predict_transformer = CutTransformer(bins = [0, 10, 20, 30, 40], labels = ["0-10", "10-20", "20-30", "30-40"]))

def build_auto_na_hist(auto_na_df, regressor, name):
	auto_na_X, auto_na_y = split_csv(auto_na_df)

	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["displacement", "horsepower", "weight", "acceleration"]] +
		[([column], [CategoricalDomain(), PMMLLabelEncoder()]) for column in ["cylinders", "model_year", "origin"]]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_na_X, auto_na_y)
	pipeline.verify(auto_na_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_na_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_na_df = load_auto("AutoNA")

	auto_na_df["cylinders"] = auto_na_df["cylinders"].astype("Int64")
	auto_na_df["model_year"] = auto_na_df["model_year"].astype("Int64")
	auto_na_df["origin"] = auto_na_df["origin"].astype("Int64")
	
	build_auto_na_hist(auto_na_df, HistGradientBoostingRegressor(max_iter = 31, categorical_features = [4, 5, 6], random_state = 13), "HistGradientBoostingAutoNA")

def build_multi_auto(auto_df, regressor, name, with_kneighbors = False):
	auto_X, auto_y = split_multi_csv(auto_df, ["acceleration", "mpg"])

	mapper = DataFrameMapper(
		[([column], ContinuousDomain()) for column in ["displacement", "horsepower", "weight"]] +
		[([column], [CategoricalDomain(), LabelBinarizer()]) for column in ["cylinders", "model_year", "origin"]]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	acceleration_mpg = DataFrame(pipeline.predict(auto_X), columns = ["acceleration", "mpg"])
	if with_kneighbors:
		Xt = pipeline_transform(pipeline, auto_X)
		kneighbors = regressor.kneighbors(Xt)
		acceleration_mpg_ids = DataFrame(kneighbors[1] + 1, columns = make_kneighbor_cols(regressor))
		acceleration_mpg = pandas.concat((acceleration_mpg, acceleration_mpg_ids), axis = 1)
	store_csv(acceleration_mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_multi_auto(auto_df, EstimatorChain([("acceleration", Link(DecisionTreeRegressor(max_depth = 3, random_state = 13), augment_funcs = ["predict", "apply"]), str(True)), ("mpg", LinearRegression(), str(True))]), "MultiEstimatorChainAuto")
	build_multi_auto(auto_df, LinearRegression(), "MultiLinearRegressionAuto")
	build_multi_auto(auto_df, KNeighborsRegressor(algorithm = "brute"), "MultiKNNAuto", with_kneighbors = True)
	build_multi_auto(auto_df, MLPRegressor(solver = "lbfgs", random_state = 13), "MultiMLPAuto")
	build_multi_auto(auto_df, MultiOutputRegressor(LinearSVR(random_state = 13)), "MultiLinearSVRAuto")
	build_multi_auto(auto_df, RegressorChain(LinearRegression()), "LinearRegressionChainAuto")

def build_housing(housing_df, regressor, name, with_kneighbors = False, **pmml_options):
	housing_X, housing_y = split_csv(housing_df)

	mapper = DataFrameMapper([
		(housing_X.columns.values, ContinuousDomain())
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("transformer-pipeline", Pipeline([
			("polynomial", PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)),
			("scaler", StandardScaler()),
			("passthrough-transformer", "passthrough"),
			("selector", SelectPercentile(score_func = f_regression, percentile = 35)),
			("passthrough-final-estimator", "passthrough")
		])),
		("regressor", regressor)
	])
	pipeline.fit(housing_X, housing_y)
	pipeline = make_pmml_pipeline(pipeline, housing_X.columns.values, housing_y.name)
	pipeline.configure(**pmml_options)
	pipeline.verify(housing_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	medv = DataFrame(pipeline.predict(housing_X), columns = ["MEDV"])
	if with_kneighbors:
		Xt = pipeline_transform(pipeline, housing_X)
		kneighbors = regressor.kneighbors(Xt)
		medv_ids = DataFrame(kneighbors[1] + 1, columns = make_kneighbor_cols(regressor))
		medv = pandas.concat((medv, medv_ids), axis = 1)
	store_csv(medv, name)

if "Housing" in datasets:
	housing_df = load_housing("Housing")

	build_housing(housing_df, AdaBoostRegressor(DecisionTreeRegressor(min_samples_leaf = 5, random_state = 13), n_estimators = 17, random_state = 13), "AdaBoostHousing")
	build_housing(housing_df, BayesianRidge(), "BayesianRidgeHousing")
	build_housing(housing_df, GBDTLMRegressor(GradientBoostingRegressor(n_estimators = 31, random_state = 13), LinearRegression()), "GBDTLMHousing")
	build_housing(housing_df, HistGradientBoostingRegressor(max_iter = 31, random_state = 13), "HistGradientBoostingHousing")
	build_housing(housing_df, KNeighborsRegressor(), "KNNHousing", with_kneighbors = True)
	build_housing(housing_df, MLPRegressor(activation = "tanh", hidden_layer_sizes = (26,), solver = "lbfgs", tol = 0.001, max_iter = 1000, random_state = 13), "MLPHousing")
	build_housing(housing_df, SGDRegressor(random_state = 13), "SGDHousing")
	build_housing(housing_df, SVR(gamma = "auto"), "SVRHousing")
	build_housing(housing_df, LinearSVR(random_state = 13), "LinearSVRHousing")
	build_housing(housing_df, NuSVR(gamma = "auto"), "NuSVRHousing")
	build_housing(housing_df, VotingRegressor([("dt", DecisionTreeRegressor(random_state = 13)), ("lr", LinearRegression())]), "VotingEnsembleHousing")

def build_visit(visit_df, regressor, name):
	visit_X, visit_y = split_csv(visit_df)

	mapper = DataFrameMapper(
		[(["edlevel"], [CategoricalDomain(), OneHotEncoder()])] +
		[([bin_column], [CategoricalDomain(), OneHotEncoder()]) for bin_column in ["outwork", "female", "married", "kids", "self"]] +
		[(["age"], ContinuousDomain())] +
		[(["hhninc", "educ"], ContinuousDomain())]
	)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(visit_X, visit_y)
	pipeline.verify(visit_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	docvis = DataFrame(pipeline.predict(visit_X), columns = ["docvis"])
	store_csv(docvis, name)

if "Visit" in datasets:
	visit_df = load_visit("Visit")

	build_visit(visit_df, GammaRegressor(), "GammaRegressionVisit")
	build_visit(visit_df, PoissonRegressor(), "PoissonRegressionVisit")

#
# Outlier detection
#

def build_ocsvm_iris(iris_df, linear_model, name):
	iris_X, iris_y = split_csv(iris_df)

	mapper = DataFrameMapper([
		(iris_X.columns.values, [ContinuousDomain(), StandardScaler()])
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("estimator", linear_model)
	])
	pipeline.fit(iris_X)
	pipeline = make_pmml_pipeline(pipeline, iris_X.columns.values)
	store_pkl(pipeline, name)
	decisionFunction = DataFrame(pipeline.decision_function(iris_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(iris_X) == -1, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name)

if "Iris" in datasets:
	build_ocsvm_iris(iris_df, SGDOneClassSVM(average = True, random_state = 13), "SGDOneClassSVMIris")

def build_iforest_housing(housing_df, iforest, name, **pmml_options):
	housing_X, housing_y = split_csv(housing_df)

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
	build_iforest_housing(housing_df, IsolationForest(contamination = 0.1, max_features = 3, random_state = 13), "IsolationForestHousing")

def build_ocsvm_housing(housing_df, svm, name):
	housing_X, housing_y = split_csv(housing_df)

	mapper = DataFrameMapper([
		(housing_X.columns.values, ContinuousDomain())
	])
	pipeline = Pipeline([
		("mapper", mapper),
		("transformer-pipeline", Pipeline([
			("none-transformer", None),
			("scaler", MaxAbsScaler()),
			("none-final-estimator", None)
		])),
		("estimator", svm)
	])
	pipeline.fit(housing_X)
	pipeline = make_pmml_pipeline(pipeline, housing_X.columns.values)
	store_pkl(pipeline, name)
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) <= 0, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name)

if "Housing" in datasets:
	build_ocsvm_housing(housing_df, OneClassSVM(gamma = "auto", nu = 0.10), "OneClassSVMHousing")
