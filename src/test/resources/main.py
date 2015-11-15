from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble.bagging import BaggingClassifier, BaggingRegressor
from sklearn.ensemble.forest import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ElasticNetCV, LassoCV
from sklearn.linear_model.ridge import RidgeCV, RidgeClassifierCV
from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Binarizer, Imputer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn_pandas import DataFrameMapper
from pandas import DataFrame

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

wheat_df = wheat_df.drop("Variety", axis = 1)

print(wheat_df.dtypes)

wheat_mapper = DataFrameMapper([
	(["Area", "Perimeter", "Compactness", "Kernel.Length", "Kernel.Width", "Asymmetry", "Groove.Length"], MinMaxScaler())
])

wheat_X = wheat_mapper.fit_transform(wheat_df)

print(wheat_X.shape)

store_pkl(wheat_mapper, "Wheat.pkl")

def kmeans_distance(kmeans, center):
	return numpy.sum(numpy.power(kmeans.cluster_centers_[center] - wheat_X, 2), axis = 1)

def build_wheat(kmeans, name, with_affinity = True):
	kmeans.fit(wheat_X)
	store_pkl(kmeans, name + ".pkl")
	cluster = DataFrame(kmeans.predict(wheat_X), columns = ["Cluster"])
	if(with_affinity == True):
		affinity_0 = kmeans_distance(kmeans, 0)
		affinity_1 = kmeans_distance(kmeans, 1)
		affinity_2 = kmeans_distance(kmeans, 2)
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

audit_mapper = DataFrameMapper([
	("Age", None),
	("Employment", LabelBinarizer()),
	("Education", LabelBinarizer()),
	("Marital", LabelBinarizer()),
	("Occupation", LabelBinarizer()),
	("Income", None),
	("Gender", LabelEncoder()),
	("Deductions", LabelEncoder()),
	("Hours", None),
	("Adjusted", None)
])

audit = audit_mapper.fit_transform(audit_df)

print(audit.shape)

store_pkl(audit_mapper, "Audit.pkl")

audit_X = audit[:, 0:48]
audit_y = audit[:, 48]

audit_y = audit_y.astype(int)

print(audit_X.dtype, audit_y.dtype)

def build_audit(classifier, name, with_proba = True):
	classifier.fit(audit_X, audit_y)
	store_pkl(classifier, name + ".pkl")
	adjusted = DataFrame(classifier.predict(audit_X), columns = ["Adjusted"])
	if(with_proba == True):
		adjusted_proba = DataFrame(classifier.predict_proba(audit_X), columns = ["probability_0", "probability_1"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name + ".csv")

build_audit(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeAudit")
build_audit(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesAudit")
build_audit(GradientBoostingClassifier(random_state = 13, loss = "exponential", init = None), "GradientBoostingAudit")
build_audit(LinearDiscriminantAnalysis(solver = "lsqr"), "LinearDiscriminantAnalysisAudit")
build_audit(LogisticRegressionCV(), "LogisticRegressionAudit")
build_audit(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleAudit")
build_audit(GaussianNB(), "NaiveBayesAudit")
build_audit(RandomForestClassifier(random_state = 13, min_samples_leaf = 5), "RandomForestAudit")
build_audit(RidgeClassifierCV(), "RidgeAudit", with_proba = False)

versicolor_df = load_csv("Versicolor.csv")

print(versicolor_df.dtypes)

versicolor_columns = versicolor_df.columns.tolist()

versicolor_mapper = DataFrameMapper([
	(versicolor_columns[: -1], RobustScaler()),
	(versicolor_columns[-1], None)
])

versicolor = versicolor_mapper.fit_transform(versicolor_df)

print(versicolor.shape)

store_pkl(versicolor_mapper, "Versicolor.pkl")

versicolor_X = versicolor[:, 0:4]
versicolor_y = versicolor[:, 4]
versicolor_y = versicolor_y.astype(int)

def build_versicolor(classifier, name, with_proba = True):
	classifier.fit(versicolor_X, versicolor_y)
	store_pkl(classifier, name + ".pkl")
	species = DataFrame(classifier.predict(versicolor_X), columns = ["Species"])
	if(with_proba == True):
		species_proba = DataFrame(classifier.predict_proba(versicolor_X), columns = ["probability_0", "probability_1"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name + ".csv")

build_versicolor(SGDClassifier(random_state = 13, n_iter = 100), "SGDVersicolor", with_proba = False)
build_versicolor(SGDClassifier(random_state = 13, loss = "log", n_iter = 100), "SGDLogVersicolor")

#
# Multi-class classification
#

iris_df = load_csv("Iris.csv")

print(iris_df.dtypes)

iris_mapper = DataFrameMapper([
	(["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [RobustScaler(), IncrementalPCA(n_components = 3, whiten = True)]),
	("Species", None)
])

iris = iris_mapper.fit_transform(iris_df)

print(iris.shape)

store_pkl(iris_mapper, "Iris.pkl")

iris_X = iris[:, 0:3]
iris_y = iris[:, 3]

print(iris_X.dtype, iris_y.dtype)

def build_iris(classifier, name, with_proba = True):
	classifier.fit(iris_X, iris_y)
	store_pkl(classifier, name + ".pkl")
	species = DataFrame(classifier.predict(iris_X), columns = ["Species"])
	if(with_proba == True):
		species_proba = DataFrame(classifier.predict_proba(iris_X), columns = ["probability_setosa", "probability_versicolor", "probability_virginica"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name + ".csv")

build_iris(DecisionTreeClassifier(random_state = 13, min_samples_leaf = 5), "DecisionTreeIris")
build_iris(ExtraTreesClassifier(random_state = 13, min_samples_leaf = 5), "ExtraTreesIris")
build_iris(GradientBoostingClassifier(random_state = 13, init = None, n_estimators = 17), "GradientBoostingIris")
build_iris(LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysisIris")
build_iris(LogisticRegressionCV(), "LogisticRegressionIris")
build_iris(BaggingClassifier(LogisticRegression(), random_state = 13, n_estimators = 3, max_features = 0.5), "LogisticRegressionEnsembleIris")
build_iris(GaussianNB(), "NaiveBayesIris")
build_iris(RandomForestClassifier(random_state = 13, min_samples_leaf = 5), "RandomForestIris")
build_iris(RidgeClassifierCV(), "RidgeIris", with_proba = False)
build_iris(SGDClassifier(random_state = 13, n_iter = 100), "SGDIris", with_proba = False)
build_iris(SGDClassifier(random_state = 13, loss = "log", n_iter = 100), "SGDLogIris")

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

auto_mapper = DataFrameMapper([
	(["cylinders"], None),
	(["displacement", "horsepower", "weight", "acceleration"], [Imputer(missing_values = "NaN"), StandardScaler()]),
	("model_year", None),
	(["origin"], Binarizer(threshold = 1)),
	("mpg", None)
])

auto = auto_mapper.fit_transform(auto_df)

print(auto.shape)

store_pkl(auto_mapper, "Auto.pkl")

auto_X = auto[:, 0:7]
auto_y = auto[:, 7]

print(auto_X.dtype, auto_y.dtype)

def build_auto(regressor, name):
	regressor = regressor.fit(auto_X, auto_y)
	store_pkl(regressor, name + ".pkl")
	mpg = DataFrame(regressor.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name + ".csv")

build_auto(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), "DecisionTreeAuto")
build_auto(ElasticNetCV(random_state = 13), "ElasticNetAuto")
build_auto(ExtraTreesRegressor(random_state = 13, min_samples_leaf = 5), "ExtraTreesAuto")
build_auto(GradientBoostingRegressor(random_state = 13, init = None), "GradientBoostingAuto")
build_auto(LassoCV(random_state = 13), "LassoAuto")
build_auto(LinearRegression(), "LinearRegressionAuto")
build_auto(BaggingRegressor(LinearRegression(), random_state = 13, max_features = 0.75), "LinearRegressionEnsembleAuto")
build_auto(RandomForestRegressor(random_state = 13, min_samples_leaf = 5), "RandomForestAuto")
build_auto(RidgeCV(), "RidgeAuto")

housing_df = load_csv("Housing.csv")

print(housing_df.dtypes)

housing_df["CHAS"] = housing_df["CHAS"].astype(float)
housing_df["RAD"] = housing_df["RAD"].astype(float)

print(housing_df.dtypes)

housing_columns = housing_df.columns.tolist()

housing_mapper = DataFrameMapper([
	(housing_columns[: -1], StandardScaler()),
	(housing_columns[-1], None)
])

housing = housing_mapper.fit_transform(housing_df)

print(housing.shape)

store_pkl(housing_mapper, "Housing.pkl")

housing_X = housing[:, 0:13]
housing_y = housing[:, 13]

def build_housing(regressor, name):
	regressor = regressor.fit(housing_X, housing_y)
	store_pkl(regressor, name + ".pkl")
	medv = DataFrame(regressor.predict(housing_X), columns = ["MEDV"])
	store_csv(medv, name + ".csv")

build_housing(SGDRegressor(random_state = 13), "SGDHousing")