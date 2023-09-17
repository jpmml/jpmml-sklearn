from pandas import DataFrame
from sklearn.base import ClassifierMixin, ClusterMixin, OutlierMixin, RegressorMixin, TransformerMixin
from sklearn2pmml import load_class_mapping
from sklearn2pmml.util import fqn

import importlib

mapping = load_class_mapping()

def resolve(k, sklearn_only = False):
	parts = k.split(".")

	if sklearn_only and parts[0] != "sklearn":
		return None

	module = ".".join(parts[0:-1])
	name = parts[-1]
	try:
		pyModule = importlib.import_module(module)
	except ModuleNotFoundError:
		return None
	try:
		pyClass = getattr(pyModule, name)
	except AttributeError:
		return None

	return pyClass

pyClasses = [resolve(k, sklearn_only = True) for k, v in mapping.items()]
#print(len(pyClasses))

_seen = {}

pyClasses = [_seen.setdefault(pyClass, pyClass) for pyClass in pyClasses if isinstance(pyClass, type) and (pyClass not in _seen)]
#print(len(pyClasses))

def make_table(pyClasses, attrs):
	columns = ["class"] + attrs
	data = []

	for pyClass in pyClasses:
		class_attrs = [fqn(pyClass)]
		for attr in attrs:
			class_attrs.append("+" if hasattr(pyClass, attr) else "-")
		data.append(class_attrs)

	return DataFrame(data = data, columns = columns)

def store_table(name, table):
	table.to_csv("attributes/" + name + ".tsv", sep = "\t", index = False)

def _is_estimator(pyClass):
	return issubclass(pyClass, (ClassifierMixin, ClusterMixin, OutlierMixin, RegressorMixin))

pyEstimatorClasses = [pyClass for pyClass in pyClasses if _is_estimator(pyClass)]
# XXX: ["n_features_", "n_features_in_", "n_outputs_"]
estimator_attrs = ["apply", "decision_function", "predict", "predict_proba"]

estimator_table = make_table(pyEstimatorClasses, estimator_attrs)
store_table("estimators", estimator_table)

def _is_transformer(pyClass):
	return issubclass(pyClass, TransformerMixin)

pyTransformerClasses = [pyClass for pyClass in pyClasses if _is_transformer(pyClass)]
# XXX: ["n_features_"]
transformer_attrs = ["transform"]

transformer_table = make_table(pyTransformerClasses, transformer_attrs)
store_table("transformers", transformer_table)
