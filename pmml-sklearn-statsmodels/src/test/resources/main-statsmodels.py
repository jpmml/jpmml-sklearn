import sys

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from main import *

from sklearn2pmml.statsmodels import StatsModelsClassifier, StatsModelsOrdinalClassifier, StatsModelsRegressor
from statsmodels.api import GLM, Logit, MNLogit, OLS, Poisson, WLS
from statsmodels.genmod import families
from statsmodels.miscmodels.ordinal_model import OrderedModel

def make_mapper(cont_cols, cat_cols):
	features = []

	if len(cont_cols) > 0:
		features.append((cont_cols, None))
	if len(cat_cols) > 0:
		features.append((cat_cols, OneHotEncoder()))

	return DataFrameMapper(features)

def build_audit(audit_df, classifier, name):
	audit_X, audit_y = split_csv(audit_df)

	cont_cols = ["Age", "Hours", "Income"]
	cat_cols = ["Deductions", "Education", "Employment", "Gender", "Marital", "Occupation"]

	mapper = make_mapper(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	classifier.remove_data()
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

audit_df = load_csv("Audit")

build_audit(audit_df, StatsModelsClassifier(GLM, family = families.Binomial()), "GLMAudit")
build_audit(audit_df, StatsModelsClassifier(Logit), "LogitAudit")

def build_iris(iris_df, classifier, name):
	iris_X, iris_y = split_csv(iris_df)

	cont_cols = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
	cat_cols = []

	mapper = make_mapper(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(iris_X, iris_y, classifier__method = "bfgs")
	classifier.remove_data()
	store_pkl(pipeline, name)
	species = DataFrame(pipeline.predict(iris_X), columns = ["Species"])
	species_proba = DataFrame(pipeline.predict_proba(iris_X), columns = ["probability(setosa)", "probability(versicolor)", "probability(virginica)"])
	species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name)

iris_df = load_iris("Iris")

build_iris(iris_df, StatsModelsClassifier(MNLogit), "MNLogitIris")

def build_auto(auto_df, regressor, name):
	auto_X, auto_y = split_csv(auto_df)

	cont_cols = ["acceleration", "displacement", "horsepower", "weight"]
	cat_cols = ["cylinders", "model_year", "origin"]

	mapper = make_mapper(cont_cols, cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(auto_X, auto_y)
	regressor.remove_data()
	store_pkl(pipeline, name)
	mpg = DataFrame(pipeline.predict(auto_X), columns = ["mpg"])
	store_csv(mpg, name)

auto_df = load_auto("Auto")

build_auto(auto_df, StatsModelsRegressor(GLM, family = families.Gaussian()), "GLMAuto")
build_auto(auto_df, StatsModelsRegressor(OLS), "OLSAuto")
build_auto(auto_df, StatsModelsRegressor(WLS), "WLSAuto")

build_auto_ordinal(auto_df, StatsModelsOrdinalClassifier(OrderedModel, distr = "logit"), "OrderedLogitAuto")

def build_visit(visit_df, regressor, name):
	visit_X, visit_y = split_csv(visit_df)

	cont_cols = ["age", "educ", "hhninc"]
	binary_cols = ["female", "kids", "married", "outwork", "self"]
	cat_cols = ["edlevel"]

	mapper = make_mapper(cont_cols, binary_cols + cat_cols)

	pipeline = PMMLPipeline([
		("mapper", mapper),
		("regressor", regressor)
	])
	pipeline.fit(visit_X, visit_y)
	regressor.remove_data()
	store_pkl(pipeline, name)
	docvis = DataFrame(pipeline.predict(visit_X), columns = ["docvis"])
	store_csv(docvis, name)

visit_df = load_visit("Visit")

build_visit(visit_df, StatsModelsRegressor(GLM, family = families.Poisson()), "GLMVisit")
build_visit(visit_df, StatsModelsRegressor(Poisson), "PoissonVisit")
