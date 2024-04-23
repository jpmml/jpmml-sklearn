import os
import sys

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn2pmml.pipeline import PMMLPipeline
from sktree.ensemble import ExtendedIsolationForest, ObliqueRandomForestClassifier, ObliqueRandomForestRegressor
from sktree.tree import ObliqueDecisionTreeClassifier, ObliqueDecisionTreeRegressor

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Housing", "Iris"]

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["Education", "Employment", "Gender", "Marital", "Occupation"]),
		("cont", StandardScaler(), ["Age", "Hours", "Income"])
	])

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	build_audit(audit_df, ObliqueDecisionTreeClassifier(criterion = "gini", splitter = "best", min_samples_leaf = 10, random_state = 13), "ObliqueDecisionTreeAudit")
	build_audit(audit_df, ObliqueRandomForestClassifier(n_estimators = 10, max_depth = 6, random_state = 13), "ObliqueRandomForestAudit")

def build_iris(iris_df, classifier, name):
	iris_X, iris_y = split_csv(iris_df)

	transformer = StandardScaler()

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	pipeline.verify(iris_X.sample(frac = 0.10, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_iris(iris_df, ObliqueDecisionTreeClassifier(criterion = "entropy", splitter = "random", min_samples_leaf = 3, random_state = 13), "ObliqueDecisionTreeIris")
	build_iris(iris_df, ObliqueRandomForestClassifier(n_estimators = 10, max_depth = 3, random_state = 13), "ObliqueRandomForestIris")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["cylinders", "model_year", "origin"]),
		("cont", StandardScaler(), ["acceleration", "displacement", "horsepower", "weight"])
	])

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	pipeline.verify(auto_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, ObliqueDecisionTreeRegressor(min_samples_leaf = 5, random_state = 13), "ObliqueDecisionTreeAuto")
	build_auto(auto_df, ObliqueRandomForestRegressor(n_estimators = 10, max_depth = 5, random_state = 13), "ObliqueRandomForestAuto")

def build_housing(housing_df, estimator, name):
	housing_X, housing_y = split_csv(housing_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["CHAS", "RAD"])
	], remainder = StandardScaler())

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("estimator", estimator)
	])
	pipeline.fit(housing_X)
	store_pkl(pipeline, name)
	decisionFunction = DataFrame(pipeline.decision_function(housing_X), columns = ["decisionFunction"])
	outlier = DataFrame(pipeline.predict(housing_X) <= 0, columns = ["outlier"]).replace(True, "true").replace(False, "false")
	store_csv(pandas.concat([decisionFunction, outlier], axis = 1), name)

if "Housing" in datasets:
	housing_df = load_housing("Housing")

	build_housing(housing_df, ExtendedIsolationForest(n_estimators = 31, contamination = 0.1, random_state = 13), "ExtendedIsolationForestHousing")
