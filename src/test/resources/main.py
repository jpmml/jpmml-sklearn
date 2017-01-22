from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame
from scipy import sparse
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
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ElasticNetCV, LassoCV
from sklearn.linear_model.ridge import RidgeCV, RidgeClassifier, RidgeClassifierCV
from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Binarizer, FunctionTransformer, Imputer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.svm import LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR
from sklearn2pmml import EstimatorProxy
from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn_pandas import DataFrameMapper
from xgboost.sklearn import XGBClassifier, XGBRegressor

import numpy
import pandas
#import pickle

def load_csv(name):
	return pandas.read_csv("csv/" + name, na_values = ["N/A", "NA"])

def store_csv(df, name):
	df.to_csv("csv/" + name, index = False)

# Joblib dump
def store_pkl(obj, name):
	joblib.dump(obj, "pkl/" + name, compress = 9)

def pipeline_transform(pipeline, X):
	identity_pipeline = Pipeline(pipeline.steps[: -1] + [("estimator", None)])
	return identity_pipeline._transform(X)

# Pickle dump
#def store_pkl(obj, name):
#	con = open("pkl/" + name, "wb")
#	pickle.dump(obj, con, protocol = -1)
#	con.close()

def dump(obj):
	for attr in dir(obj):
		print("obj.%s = %s" % (attr, getattr(obj, attr)))

#
# Clustering
#

wheat_df = load_csv("Wheat.csv")

print(wheat_df.dtypes)

wheat_X = wheat_df[["Area", "Perimeter", "Compactness", "Kernel.Length", "Kernel.Width", "Asymmetry", "Groove.Length"]]
wheat_y = wheat_df["Variety"]

def kmeans_distance(kmeans, center, X):
	return numpy.sum(numpy.power(kmeans.cluster_centers_[center] - X, 2), axis = 1)

def build_wheat(kmeans, name, with_affinity = True):
	mapper = DataFrameMapper([
		(["Area", "Perimeter", "Compactness", "Kernel.Length", "Kernel.Width", "Asymmetry", "Groove.Length"], ContinuousDomain())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("transformer", FunctionTransformer(numpy.log10)),
		("scaler", MinMaxScaler()),
		("clusterer", kmeans)
	])
	pipeline.fit(wheat_X)
	store_pkl(pipeline, name + ".pkl")
	cluster = DataFrame(pipeline.predict(wheat_X), columns = ["Cluster"])
	if(with_affinity == True):
		Xt = pipeline_transform(pipeline, wheat_X)
		affinity_0 = kmeans_distance(kmeans, 0, Xt)
		affinity_1 = kmeans_distance(kmeans, 1, Xt)
		affinity_2 = kmeans_distance(kmeans, 2, Xt)
		cluster_affinity = DataFrame(numpy.transpose([affinity_0, affinity_1, affinity_2]), columns = ["affinity_0", "affinity_1", "affinity_2"])
		cluster = pandas.concat((cluster, cluster_affinity), axis = 1)
	store_csv(cluster, name + ".csv")

build_wheat(KMeans(n_clusters = 3, random_state = 13), "KMeansWheat")
build_wheat(MiniBatchKMeans(n_clusters = 3, compute_labels = False, random_state = 13), "MiniBatchKMeansWheat")

#
# Binary classification
#

audit_df = load_csv("Audit.csv")

print(audit_df.dtypes)

audit_df["Deductions"] = audit_df["Deductions"].replace(True, "TRUE").replace(False, "FALSE").astype(str)

print(audit_df.dtypes)

audit_X = audit_df[["Age", "Employment", "Education", "Marital", "Occupation", "Income", "Gender", "Deductions", "Hours"]]
audit_y = audit_df["Adjusted"]
audit_y = audit_y.astype(int)

def build_audit(classifier, name, with_proba = True):
	mapper = DataFrameMapper([
		("Age", ContinuousDomain()),
		("Employment", [LabelBinarizer(), SelectFromModel(EstimatorProxy(DecisionTreeClassifier(random_state = 13)), threshold = "1.25 * mean")]),
		("Education", [LabelBinarizer(), SelectFromModel(EstimatorProxy(RandomForestClassifier(random_state = 13, n_estimators = 3)), threshold = "median")]),
		("Marital", [LabelBinarizer(), SelectKBest(k = 3)]),
		("Occupation", [LabelBinarizer(), SelectKBest(k = 3)]),
		("Income", ContinuousDomain()),
		("Gender", LabelEncoder()),
		("Deductions", LabelEncoder()),
		("Hours", ContinuousDomain())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name + ".pkl")
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	if(with_proba == True):
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability_0", "probability_1"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name + ".csv")

build_audit(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeAudit")
build_audit(BaggingClassifier(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleAudit")
build_audit(DummyClassifier(strategy = "most_frequent"), "DummyAudit")
build_audit(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesAudit")
build_audit(GradientBoostingClassifier(random_state = 13, loss = "exponential", init = None), "GradientBoostingAudit")
build_audit(LGBMClassifier(seed = 13, objective = "binary"), "LGBMAudit")
build_audit(LinearDiscriminantAnalysis(solver = "lsqr"), "LinearDiscriminantAnalysisAudit")
build_audit(LogisticRegressionCV(), "LogisticRegressionAudit")
build_audit(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleAudit")
build_audit(GaussianNB(), "NaiveBayesAudit")
build_audit(RandomForestClassifier(random_state = 13, min_samples_leaf = 5), "RandomForestAudit")
build_audit(RidgeClassifierCV(), "RidgeAudit", with_proba = False)
build_audit(BaggingClassifier(RidgeClassifier(random_state = 13), random_state = 13, n_estimators = 3, max_features = 0.5), "RidgeEnsembleAudit")
build_audit(SVC(), "SVCAudit", with_proba = False)
build_audit(VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression())], voting = "soft", weights = [3, 1, 2]), "VotingEnsembleAudit")
build_audit(XGBClassifier(objective = "binary:logistic"), "XGBAudit")

versicolor_df = load_csv("Versicolor.csv")

print(versicolor_df.dtypes)

versicolor_columns = versicolor_df.columns.tolist()

versicolor_X = versicolor_df[versicolor_columns[: -1]]
versicolor_y = versicolor_df[versicolor_columns[-1]]
versicolor_y = versicolor_y.astype(int)

def build_versicolor(classifier, name, with_proba = True):
	mapper = DataFrameMapper([
		((versicolor_columns[: -1], [ContinuousDomain(), RobustScaler()]))
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("transformer", PolynomialFeatures(degree = 3)),
		("selector", SelectKBest(k = "all")),
		("classifier", classifier)
	])
	pipeline.fit(versicolor_X, versicolor_y)
	store_pkl(pipeline, name + ".pkl")
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	if(with_proba == True):
		species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability_0", "probability_1"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name + ".csv")

build_versicolor(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeVersicolor")
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

iris_df = load_csv("Iris.csv")

print(iris_df.dtypes)

iris_X = iris_df[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]]
iris_y = iris_df["Species"]

def build_iris(classifier, name, with_proba = True):
	mapper = DataFrameMapper([
		(["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], ContinuousDomain())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("scaler", RobustScaler()),
		("pca", IncrementalPCA(n_components = 3, whiten = True)),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name + ".pkl")
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	if(with_proba == True):
		species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability_setosa", "probability_versicolor", "probability_virginica"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name + ".csv")

build_iris(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeIris")
build_iris(BaggingClassifier(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleIris")
build_iris(DummyClassifier(strategy = "constant", constant = "versicolor"), "DummyIris")
build_iris(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesIris")
build_iris(GradientBoostingClassifier(random_state = 13, init = None, n_estimators = 17), "GradientBoostingIris")
build_iris(KNeighborsClassifier(), "KNNIris", with_proba = False)
build_iris(LGBMClassifier(seed = 13, objective = "multiclass"), "LGBMIris")
build_iris(LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysisIris")
build_iris(LogisticRegressionCV(), "LogisticRegressionIris")
build_iris(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleIris")
build_iris(MLPClassifier(hidden_layer_sizes = (6,), solver = "lbfgs", random_state = 13, tol = 0.1, max_iter = 100), "MLPIris")
build_iris(GaussianNB(), "NaiveBayesIris")
build_iris(RandomForestClassifier(random_state = 13, min_samples_leaf = 5), "RandomForestIris")
build_iris(RidgeClassifierCV(), "RidgeIris", with_proba = False)
build_iris(BaggingClassifier(RidgeClassifier(random_state = 13), random_state = 13, n_estimators = 3, max_features = 0.5), "RidgeEnsembleIris")
build_iris(SGDClassifier(random_state = 13, n_iter = 100), "SGDIris", with_proba = False)
build_iris(SGDClassifier(random_state = 13, loss = "log", n_iter = 100), "SGDLogIris")
build_iris(SVC(), "SVCIris", with_proba = False)
build_iris(NuSVC(), "NuSVCIris", with_proba = False)
build_iris(VotingClassifier([("dt", DecisionTreeClassifier(random_state = 13)), ("nb", GaussianNB()), ("lr", LogisticRegression())]), "VotingEnsembleIris", with_proba = False)
build_iris(XGBClassifier(objective = "multi:softmax"), "XGBIris")

#
# Regression
#

auto_df = load_csv("Auto.csv")

print(auto_df.dtypes)

auto_df["displacement"] = auto_df["displacement"].astype(float)
auto_df["horsepower"] = auto_df["horsepower"].astype(float)
auto_df["weight"] = auto_df["weight"].astype(float)
auto_df["acceleration"] = auto_df["acceleration"].astype(float)

print(auto_df.dtypes)

auto_X = auto_df[["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]]
auto_y = auto_df["mpg"]

def build_auto(regressor, name):
	mapper = DataFrameMapper([
		(["cylinders"], CategoricalDomain()),
		(["displacement", "horsepower", "weight", "acceleration"], [ContinuousDomain(), Imputer(missing_values = "NaN"), StandardScaler()]),
		(["model_year"], [CategoricalDomain(), Binarizer(threshold = 77)]), # Pre/post 1973 oil crisis effects
		(["origin"], OneHotEncoder())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name + ".pkl")
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name + ".csv")

build_auto(AdaBoostRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 17), "AdaBoostAuto")
build_auto(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), "DecisionTreeAuto")
build_auto(BaggingRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleAuto")
build_auto(DummyRegressor(strategy = "median"), "DummyAuto")
build_auto(ElasticNetCV(random_state = 13), "ElasticNetAuto")
build_auto(ExtraTreesRegressor(random_state = 13, min_samples_leaf = 5), "ExtraTreesAuto")
build_auto(GradientBoostingRegressor(random_state = 13, init = None), "GradientBoostingAuto")
build_auto(LassoCV(random_state = 13), "LassoAuto")
build_auto(LGBMRegressor(seed = 13, n_estimators = 17), "LGBMAuto")
build_auto(LinearRegression(), "LinearRegressionAuto")
build_auto(BaggingRegressor(LinearRegression(), random_state = 13, max_features = 0.5), "LinearRegressionEnsembleAuto")
build_auto(RandomForestRegressor(random_state = 13, min_samples_leaf = 5), "RandomForestAuto")
build_auto(RidgeCV(), "RidgeAuto")
build_auto(XGBRegressor(objective = "reg:linear"), "XGBAuto")

housing_df = load_csv("Housing.csv")

print(housing_df.dtypes)

housing_df["CHAS"] = housing_df["CHAS"].astype(float)
housing_df["RAD"] = housing_df["RAD"].astype(float)

print(housing_df.dtypes)

housing_columns = housing_df.columns.tolist()

housing_X = housing_df[housing_columns[: -1]]
housing_y = housing_df[housing_columns[-1]]

def build_housing(regressor, name, with_kneighbors = False):
	mapper = DataFrameMapper([
		(housing_columns[: -1], ContinuousDomain())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("transformer", PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)),
		("scaler", StandardScaler()),
		("selector", SelectKBest(k = 7)),
		("regressor", regressor)
	])
	pipeline.fit(housing_X, housing_y)
	store_pkl(pipeline, name + ".pkl")
	medv = DataFrame(pipeline.predict(housing_X), columns = ["MEDV"])
	if(with_kneighbors == True):
		Xt = pipeline_transform(pipeline, housing_X)
		kneighbors = regressor.kneighbors(Xt)
		medv_ids = DataFrame(kneighbors[1] + 1, columns = ["neighbor_" + str(x + 1) for x in range(regressor.n_neighbors)])
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

def build_iforest_housing_anomaly(iforest, name):
	mapper = DataFrameMapper([
		(housing_columns[: -1], ContinuousDomain())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("estimator", Pipeline([("first", iforest)]))
	])
	pipeline.fit(housing_X)
	store_pkl(pipeline, name + ".pkl")
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) == -1, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name + ".csv")

build_iforest_housing_anomaly(IsolationForest(random_state = 13), "IsolationForestHousingAnomaly")

def build_svm_housing_anomaly(svm, name):
	mapper = DataFrameMapper([
		(housing_columns[: -1], ContinuousDomain())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("estimator", Pipeline([("first", MaxAbsScaler()), ("second", svm)]))
	])
	pipeline.fit(housing_X)
	store_pkl(pipeline, name + ".pkl")
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) <= 0, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name + ".csv")

build_svm_housing_anomaly(OneClassSVM(nu = 0.10, random_state = 13), "OneClassSVMHousingAnomaly")
