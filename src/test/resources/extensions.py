from common import *

from pandas import DataFrame
from sklearn2pmml import make_tpot_pmml_config, make_pmml_pipeline
from tpot import TPOTClassifier, TPOTRegressor
from tpot.config import classifier_config_dict, regressor_config_dict

import pandas

#
# Classification
#

def build_classifier(data, name):
	X, y = data
	categories = pandas.unique(y)
	config = make_tpot_pmml_config(classifier_config_dict)
	del config["sklearn.neighbors.KNeighborsClassifier"]
	classifier = TPOTClassifier(generations = 1, population_size = 3, random_state = 13, config_dict = config, verbosity = 2)
	classifier.fit(X, y)
	pipeline = make_pmml_pipeline(classifier.fitted_pipeline_, active_fields = X.columns.values, target_fields = [y.name])
	print(repr(pipeline))
	store_pkl(pipeline, name + ".pkl")
	result = DataFrame(classifier.predict(X), columns = [y.name])
	if(len(categories) > 0):
		probabilities = DataFrame(classifier.predict_proba(X), columns = ["probability(" + str(category) + ")" for category in categories])
		result = pandas.concat([result, probabilities], axis = 1)
	store_csv(result, name + ".csv")

iris_data = load_iris("Iris.csv")

build_classifier(iris_data, "TPOTIris")

versicolor_data = load_iris("Versicolor.csv")

build_classifier(versicolor_data, "TPOTVersicolor")

#
# Regression
#

def build_regressor(data, name):
	X, y = data
	config = make_tpot_pmml_config(regressor_config_dict)
	del config["sklearn.neighbors.KNeighborsRegressor"]
	regressor = TPOTRegressor(generations = 3, population_size = 3, random_state = 13, config_dict = config, verbosity = 2)
	regressor.fit(X, y)
	pipeline = make_pmml_pipeline(regressor.fitted_pipeline_, active_fields = X.columns.values, target_fields = [y.name])
	print(repr(pipeline))
	store_pkl(pipeline, name + ".pkl")
	result = DataFrame(regressor.predict(X), columns = [y.name])
	store_csv(result, name + ".csv")

auto_data = load_auto("Auto.csv")

build_regressor(auto_data, "TPOTAuto")

housing_data = load_housing("Housing.csv")

build_regressor(housing_data, "TPOTHousing")
