import os
import sys

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline
from sktree.tree import ObliqueDecisionTreeClassifier, ObliqueDecisionTreeRegressor

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris"]

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["Education", "Employment", "Gender", "Marital", "Occupation"]),
		("cont", "passthrough", ["Age", "Hours", "Income"])
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

def build_iris(iris_df, classifier, name):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = PMMLPipeline([
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

	build_iris(iris_df, ObliqueDecisionTreeClassifier(criterion = "entropy", splitter = "random", max_depth = 3, random_state = 13), "ObliqueDecisionTreeIris")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	transformer = ColumnTransformer([
		("cat", OneHotEncoder(sparse_output = False), ["cylinders", "model_year", "origin"]),
		("cont", "passthrough", ["acceleration", "displacement", "horsepower", "weight"])
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