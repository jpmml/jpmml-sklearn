from common import *

from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn2pmml import make_tpot_pmml_config, make_pmml_pipeline
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import ConcatTransformer, ExpressionTransformer
from tpot import TPOTClassifier, TPOTRegressor
from tpot.config import classifier_config_dict, regressor_config_dict

import pandas

def filter_config(config):
	return { key: value for key, value in config.items() if not (key.startswith("sklearn.ensemble.") or key.startswith("xgboost.")) }

#
# Classification
#

def build_classifier(data, feature_pipeline, generations, population_size, name):
	X, y = data
	Xt = feature_pipeline.fit_transform(X)
	Xt = Xt.astype(float)
	categories = pandas.unique(y)
	config = make_tpot_pmml_config(classifier_config_dict)
	config = filter_config(config)
	del config["sklearn.naive_bayes.GaussianNB"] # Does not support nesting - see http://mantis.dmg.org/view.php?id=208
	del config["sklearn.neighbors.KNeighborsClassifier"]
	del config["sklearn.svm.LinearSVC"] # Does not support classifier.predict_proba(Xt)
	del config["sklearn.tree.DecisionTreeClassifier"]
	classifier = TPOTClassifier(generations = generations, population_size = population_size, random_state = 13, config_dict = config, verbosity = 2)
	classifier.fit(Xt, y)
	pipeline = Pipeline(steps = feature_pipeline.steps + classifier.fitted_pipeline_.steps)
	pipeline = make_pmml_pipeline(pipeline, active_fields = X.columns.values, target_fields = [y.name])
	pipeline.verify(X.sample(frac = 0.05, random_state = 13))
	print(repr(pipeline))
	store_pkl(pipeline, name)
	result = DataFrame(classifier.predict(Xt), columns = [y.name])
	if(len(categories) > 0):
		probabilities = DataFrame(classifier.predict_proba(Xt), columns = ["probability(" + str(category) + ")" for category in categories])
		result = pandas.concat([result, probabilities], axis = 1)
	store_csv(result, name)

audit_data = load_audit("Audit")

audit_feature_pipeline = Pipeline([
	("mapper", DataFrameMapper(
		[(cat_column, [CategoricalDomain(), LabelBinarizer()]) for cat_column in ["Employment", "Education", "Marital", "Occupation", "Gender"]] +
		[(cont_column, ContinuousDomain()) for cont_column in ["Age", "Income", "Hours"]] +
		[(["Income", "Hours"], Alias(ExpressionTransformer("X[0] / (X[1] * 52.0)"), "Hourly_Income", prefit = True))]
	, df_out = True))
])

build_classifier(audit_data, audit_feature_pipeline, 3, 7, "TPOTAudit")

iris_data = load_iris("Iris")

iris_feature_pipeline = Pipeline([
	("mapper", DataFrameMapper([
		(iris_data[0].columns.values, ContinuousDomain())
	]))
])

build_classifier(iris_data, iris_feature_pipeline, 7, 17, "TPOTIris")

versicolor_data = load_iris("Versicolor")

build_classifier(versicolor_data, iris_feature_pipeline, 7, 17, "TPOTVersicolor")

#
# Regression
#

def build_regressor(data, feature_pipeline, generations, population_size, name):
	X, y = data
	Xt = feature_pipeline.fit_transform(X)
	Xt = Xt.astype(float)
	config = make_tpot_pmml_config(regressor_config_dict)
	config = filter_config(config)
	del config["sklearn.neighbors.KNeighborsRegressor"]
	regressor = TPOTRegressor(generations = generations, population_size = population_size, random_state = 13, config_dict = config, verbosity = 2)
	regressor.fit(Xt, y)
	pipeline = Pipeline(steps = feature_pipeline.steps + regressor.fitted_pipeline_.steps)
	pipeline = make_pmml_pipeline(pipeline, active_fields = X.columns.values, target_fields = [y.name])
	pipeline.verify(X.sample(frac = 0.05, random_state = 13))
	print(repr(pipeline))
	store_pkl(pipeline, name)
	result = DataFrame(regressor.predict(Xt), columns = [y.name])
	store_csv(result, name)

auto_data = load_auto("Auto")

auto_feature_pipeline = Pipeline([
	("mapper", DataFrameMapper(
		[(cat_column, [CategoricalDomain(), LabelBinarizer()]) for cat_column in ["cylinders", "model_year", "origin"]] +
		# XXX
		#[(["cylinders", "origin"], [ConcatTransformer("/"), LabelBinarizer()])] +
		[(cont_column, ContinuousDomain()) for cont_column in ["displacement", "horsepower", "weight", "acceleration"]]
	))
])

build_regressor(auto_data, auto_feature_pipeline, 7, 17, "TPOTAuto")

housing_data = load_housing("Housing")

housing_feature_pipeline = Pipeline([
	("mapper", DataFrameMapper([
		(housing_data[0].columns.values, ContinuousDomain())
	]))
])

build_regressor(housing_data, housing_feature_pipeline, 5, 11, "TPOTHousing")
