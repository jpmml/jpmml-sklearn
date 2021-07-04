from common import *

from mlxtend.preprocessing import DenseTransformer
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import ExpressionTransformer
from sklego.meta import EstimatorTransformer
from sklego.preprocessing import IdentityTransformer

def make_estimator_transformer(estimator, pmml_name):
	estimator.pmml_name_ = pmml_name
	return EstimatorTransformer(estimator)

def make_estimatortransformer_pipeline(cat_cols, cont_cols, transformer_estimator, final_estimator):
	cat_encoder = Pipeline([
		("encoder", OneHotEncoder()),
		("formatter", DenseTransformer()),
		("transformer", make_estimator_transformer(transformer_estimator, "transformer"))
	])
	cont_encoder = IdentityTransformer()
	pipeline = PMMLPipeline([
		("mapper", ColumnTransformer([
			("cat", cat_encoder, cat_cols),
			("cont", cont_encoder, cont_cols)
		])),
		("estimator", final_estimator)
	])
	return pipeline

def build_estimatortransformer_audit(name):
	audit_X, audit_y = load_audit("Audit")

	pipeline = make_estimatortransformer_pipeline(["Employment", "Education", "Marital", "Occupation", "Gender"], ["Age", "Income", "Hours"], LogisticRegression(random_state = 13), DecisionTreeClassifier(min_samples_split = 10, random_state = 13))
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

build_estimatortransformer_audit("EstimatorTransformerAudit")

def build_estimatortransformer_iris(outlier_detector_estimator, final_estimator, name):
	iris_X, iris_y = load_iris("Iris")

	pipeline = PMMLPipeline([
		("decorator", ContinuousDomain()),
		("outlier_detector", FeatureUnion([
			("original", IdentityTransformer()),
			("flag", make_estimator_transformer(outlier_detector_estimator, "outlierDetector")),	
		])),
		("estimator", final_estimator)
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

build_estimatortransformer_iris(IsolationForest(n_estimators = 3, max_features = 2, contamination = 0.03, random_state = 13), LogisticRegression(random_state = 13), "EstimatorTransformerIris")

def build_estimatortransformer_auto(name):
	auto_X, auto_y = load_auto("Auto")

	pipeline = make_estimatortransformer_pipeline(["cylinders", "model_year", "origin"], ["displacement", "horsepower", "weight", "acceleration"], LinearRegression(), DecisionTreeRegressor(min_samples_split = 25, random_state = 13))
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

build_estimatortransformer_auto("EstimatorTransformerAuto")