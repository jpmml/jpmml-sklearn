from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours, NearMiss, OneSidedSelection, RandomUnderSampler, TomekLinks
from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import ExpressionTransformer

import os
import sys

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

from common import *

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	mapper = DataFrameMapper(
		[(cat_column, [CategoricalDomain(), LabelBinarizer()]) for cat_column in ["Employment", "Education", "Marital", "Occupation", "Gender"]] +
		[(cont_column, ContinuousDomain()) for cont_column in ["Age", "Income", "Hours"]] +
		[(["Income", "Hours"], Alias(ExpressionTransformer("X[0] / (X[1] * 52.0)"), "Hourly_Income", prefit = True))]
	, df_out = True)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

audit_df = load_audit("Audit")

build_audit(audit_df, BalancedBaggingClassifier(n_estimators = 3, random_state = 13), "BalancedDecisionTreeEnsembleAudit")
build_audit(audit_df, BalancedRandomForestClassifier(n_estimators = 10, random_state = 13), "BalancedRandomForestAudit")

def build_iris(iris_df, sampler, name):
	iris_X, iris_y = split_csv(iris_df)

	pipeline = PMMLPipeline([
		("pipeline", make_pipeline(ContinuousDomain(), sampler, LogisticRegression(multi_class = "ovr", random_state = 13)))
	])
	pipeline.fit(iris_X, iris_y)
	pipeline.verify(iris_X.sample(frac = 0.1, random_state = 13))
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

iris_df = load_iris("Iris")

iris_over_sample = {"setosa" : 50, "versicolor" : 75, "virginica" : 50}
iris_under_sample = {"setosa" : 25, "versicolor" : 50, "virginica" : 25}

build_iris(iris_df, ADASYN(sampling_strategy = iris_over_sample, random_state = 13), "ADASYNIris")
build_iris(iris_df, ClusterCentroids(sampling_strategy = iris_under_sample, estimator = KMeans(n_clusters = 5, random_state = 13), random_state = 13), "ClusterCentroidsIris")
build_iris(iris_df, NearMiss(sampling_strategy = iris_under_sample), "NearMissIris")
build_iris(iris_df, OneSidedSelection(sampling_strategy = ["setosa"], random_state = 13), "OneSidedSelectionIris")
build_iris(iris_df, RandomOverSampler(sampling_strategy = iris_over_sample, random_state = 13), "RandomOverSamplerIris")
build_iris(iris_df, RandomUnderSampler(sampling_strategy = iris_under_sample, random_state = 13), "RandomUnderSamplerIris")
build_iris(iris_df, SMOTE(sampling_strategy = iris_over_sample, random_state = 13), "SMOTEIris")
build_iris(iris_df, SMOTEENN(smote = SMOTE(sampling_strategy = iris_over_sample, random_state = 13), enn = EditedNearestNeighbours(), random_state = 13), "SMOTEENNIris")
build_iris(iris_df, SMOTETomek(smote = SMOTE(sampling_strategy = iris_over_sample, random_state = 13), tomek = TomekLinks(sampling_strategy = ["versicolor", "virginica"]), random_state = 13), "SMOTETomekIris")
build_iris(iris_df, TomekLinks(sampling_strategy = ["versicolor", "virginica"]), "TomekLinksIris")
