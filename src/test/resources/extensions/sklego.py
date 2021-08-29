from common import *

from mlxtend.preprocessing import DenseTransformer
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import ExpressionTransformer
from sklego.meta import EstimatorTransformer
from sklego.preprocessing import IdentityTransformer

def make_estimator_transformer(estimator, pmml_name, predict_func = "predict"):
	estimator.pmml_name_ = pmml_name
	return EstimatorTransformer(estimator, predict_func = predict_func)

def make_enhancer(estimator_transformer):
	return FeatureUnion([
		("identity", IdentityTransformer()),
		("estimator_transformer", estimator_transformer)
	])

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

def build_estimatortransformer_wheat(name):
	wheat_X, wheat_y = load_wheat("Wheat")

	pipeline = PMMLPipeline([
		("decorator", ContinuousDomain()),
		("cluster_labeler", make_enhancer(make_pipeline(make_estimator_transformer(KMeans(n_clusters = 5, random_state = 13), "clusterLabeler"), OneHotEncoder(sparse = False)))),
		("classifier", LogisticRegression(random_state = 13))
	])
	pipeline.fit(wheat_X, wheat_y)
	store_pkl(pipeline, name)
	variety = DataFrame(pipeline.predict(wheat_X), columns = ["Variety"])
	variety_proba = DataFrame(pipeline.predict_proba(wheat_X), columns = ["probability(1)", "probability(2)", "probability(3)"])
	variety = pandas.concat((variety, variety_proba), axis = 1)
	store_csv(variety, name)

build_estimatortransformer_wheat("EstimatorTransformerWheat")

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

def build_estimatortransformer_versicolor(outlier_detector_estimator, final_estimator, name):
	versicolor_X, versicolor_y = load_versicolor("Versicolor")

	pipeline = PMMLPipeline([
		("decorator", ContinuousDomain()),
		("outlier_detector", make_enhancer(make_pipeline(make_estimator_transformer(outlier_detector_estimator, "outlierDetector"), OneHotEncoder(sparse = False)))),
		("estimator", final_estimator)
	])
	pipeline.fit(versicolor_X, versicolor_y)
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(versicolor_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(versicolor_X), columns = ["probability(0)", "probability(1)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

build_estimatortransformer_versicolor(OneClassSVM(), LinearDiscriminantAnalysis(), "EstimatorTransformerVersicolor")

def build_estimatortransformer_iris(outlier_detector_estimator, final_estimator, name):
	iris_X, iris_y = load_iris("Iris")

	pipeline = PMMLPipeline([
		("decorator", ContinuousDomain()),
		("outlier_detector", make_enhancer(make_estimator_transformer(outlier_detector_estimator, "outlierDetector", predict_func = "decision_function"))),
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

def build_estimatortransformer_housing(name):
	housing_X, housing_y = load_housing("Housing")

	pipeline = PMMLPipeline([
		("decorator", ContinuousDomain()),
		("leaf_labeler", make_enhancer(make_pipeline(make_estimator_transformer(DecisionTreeRegressor(max_depth = 3, min_samples_leaf = 20, random_state = 13), "leafLabeler", predict_func = "apply"), OneHotEncoder(sparse = False)))),
		("estimator", LinearRegression())
	])
	pipeline.fit(housing_X, housing_y)
	store_pkl(pipeline, name)
	medv = DataFrame(pipeline.predict(housing_X), columns = ["MEDV"])
	store_csv(medv, name)

build_estimatortransformer_housing("EstimatorTransformerHousing")