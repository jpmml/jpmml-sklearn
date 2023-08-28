import os
import sys

from hpsklearn import hist_gradient_boosting_classifier, linear_regression, logistic_regression, pca
from hpsklearn import HyperoptEstimator
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris"]

def to_pipeline(estimator):
	steps = []
	steps += estimator._best_preprocs
	steps += [estimator._best_learner]

	return make_pipeline(*steps)

def build_audit(audit_df, name):
	audit_X, audit_y = split_csv(audit_df)

	mapper = ColumnTransformer([
		("cat", OrdinalEncoder(dtype = int), [1, 2, 3, 4, 6]),
		("cont", "passthrough", [])
	], remainder = "drop")

	estimator = HyperoptEstimator(classifier = hist_gradient_boosting_classifier("hist", categorical_features = [0, 1, 2, 3, 4]), preprocessing = [mapper])
	estimator.fit(audit_X, audit_y)
	mapper.feature_names_in_ = audit_X.columns.values
	store_pkl(estimator, name)

	pipeline = to_pipeline(estimator)

	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["y"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	build_audit(audit_df, "HyperoptAudit")

def build_iris(iris_df, name):
	iris_X, iris_y = split_csv(iris_df)

	estimator = HyperoptEstimator(classifier = logistic_regression("lr"), preprocessing = [pca("pca", n_components = 3)])
	estimator.fit(iris_X, iris_y)
	store_pkl(estimator, name)

	pipeline = to_pipeline(estimator)

	species = DataFrame(pipeline.predict(iris_X), columns = ["y"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_iris(iris_df, "HyperoptIris")

def build_auto(auto_df, name):
	auto_X, auto_y = split_csv(auto_df)

	mapper = ColumnTransformer([
		("cat", OneHotEncoder(handle_unknown = "infrequent_if_exist"), [0, 5, 6])
	], remainder = StandardScaler())

	estimator = HyperoptEstimator(regressor = linear_regression("lm"), preprocessing = [mapper])
	estimator.fit(auto_X, auto_y)
	mapper.feature_names_in_ = auto_X.columns.values
	store_pkl(estimator, name)

	mpg = DataFrame(estimator.predict(auto_X), columns = ["y"])
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, "HyperoptAuto")