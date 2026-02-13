import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from common import *

from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import k_categorical, Bernoulli, LogNormal, Normal, Poisson
from ngboost.scores import LogScore, MLE
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import numpy

numpy.random.seed(13)

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris", "Visit"]

def make_column_transformer(cat_cols, cont_cols):
	return ColumnTransformer(
		[(cat_col, make_pipeline(CategoricalDomain(), OneHotEncoder()), [cat_col]) for cat_col in cat_cols] +
		[(cont_col, ContinuousDomain(), [cont_col]) for cont_col in cont_cols]
	)

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
	cont_cols = ["Age", "Hours", "Income"]

	transformer = make_column_transformer(cat_cols, cont_cols)

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	# XXX
	classifier.fitted_ = True
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	build_audit(audit_df, NGBClassifier(Dist = Bernoulli, n_estimators = 31, learning_rate = 0.1, Score = LogScore, random_state = 13), "NGBoostAudit")

def build_iris(iris_df, classifier, name):
	iris_X, iris_y = split_csv(iris_df)

	le = LabelEncoder()
	iris_y = Series(le.fit_transform(iris_y), name = "Species")

	pipeline = PMMLPipeline([
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y)
	store_pkl(pipeline, name)
	# XXX
	classifier.fitted_ = True
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(0)", "probability(1)", "probability(2)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

if "Iris" in datasets:
	iris_df = load_iris("Iris")

	build_iris(iris_df, NGBClassifier(Dist = k_categorical(3), n_estimators = 5, learning_rate = 0.1, Score = LogScore, random_state = 13), "NGBoostIris")

def build_auto(auto_df, regressor, name, ci = None):
	auto_X, auto_y = split_csv(auto_df)

	cat_cols = ["cylinders", "model_year", "origin"]
	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

	transformer = make_column_transformer(cat_cols, cont_cols)

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	if ci:
		pipeline.configure(confidence_level = (ci if name == "NGBoostAuto" else True))
	store_pkl(pipeline, name)
	# XXX
	regressor.fitted_ = True
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	if ci:
		auto_Xt = transformer.transform(auto_X)
		dist = regressor.pred_dist(auto_Xt)
		mpg["lower(mpg)"] = dist.ppf((1 - ci) / 2)
		mpg["upper(mpg)"] = dist.ppf(1 - (1 - ci) / 2)
	store_csv(mpg, name)

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	build_auto(auto_df, NGBRegressor(Dist = Normal, n_estimators = 17, learning_rate = 0.1, Score = MLE, random_state = 13), "NGBoostAuto", ci = 0.95)
	build_auto(auto_df, NGBRegressor(Dist = LogNormal, n_estimators = 17, learning_rate = 0.1, Score = LogScore, random_state = 13), "NGBoostLogAuto", ci = 0.95)

def build_visit(visit_df, regressor, name, ci = None):
	visit_X, visit_y = split_csv(visit_df)

	cat_cols = ["edlevel"] + ["female", "kids", "married", "outwork"]
	cont_cols = ["age", "educ", "hhninc"]

	transformer = make_column_transformer(cat_cols, cont_cols)

	pipeline = PMMLPipeline([
		("transformer", transformer),
		("regressor", regressor)
	])
	pipeline.fit(visit_X, visit_y)
	store_pkl(pipeline, name)
	# XXX
	regressor.fitted_ = True
	docvis = DataFrame(pipeline.predict(visit_X), columns = ["docvis"])
	store_csv(docvis, name)

if "Visit" in datasets:
	visit_df = load_visit("Visit")

	build_visit(visit_df, NGBRegressor(Dist = Poisson, n_estimators = 31, learning_rate = 0.1, Score = LogScore, random_state = 13), "NGBoostVisit")
