from common import *

from mlxtend.preprocessing import DenseTransformer
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklego.meta import EstimatorTransformer
from sklego.preprocessing import IdentityTransformer

def make_estimatortransformer_pipeline(cat_cols, cont_cols, transformer_estimator, final_estimator):
	cat_encoder = Pipeline([
		("encoder", OneHotEncoder()),
		("formatter", DenseTransformer()),
		("estimator", EstimatorTransformer(transformer_estimator))
	])
	cont_encoder = IdentityTransformer()
	pipeline = PMMLPipeline([
		("mapper", ColumnTransformer([
			("cat", cat_encoder, cat_cols),
			("cont", cont_encoder, cont_cols)
		])),
		("classifier", final_estimator)
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

def build_estimatortransformer_auto(name):
	auto_X, auto_y = load_auto("Auto")

	pipeline = make_estimatortransformer_pipeline(["cylinders", "model_year", "origin"], ["displacement", "horsepower", "weight", "acceleration"], LinearRegression(), DecisionTreeRegressor(min_samples_split = 25, random_state = 13))
	pipeline.fit(auto_X, auto_y)
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

build_estimatortransformer_auto("EstimatorTransformerAuto")