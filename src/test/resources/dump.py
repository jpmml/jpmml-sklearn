from sklearn.datasets import load_iris
from sklearn.externals import joblib as sklearn_joblib
from sklearn.linear_model import LogisticRegressionCV

import joblib
import pickle
import platform

def _platform():
	version = platform.python_version_tuple()
	return ("python-" + version[0] + "." + version[1])

def _module(name, version):
	return (name + "-" + version)

def _platform_module(name, version):
	return (_platform() + "-" + _module(name, version))

def _pickle(protocol):
	con = open("dump/" + _platform_module("pickle", "p" + str(protocol)) + ".pkl", "wb")
	pickle.dump(iris_classifier, con, protocol = protocol)
	con.close()

iris = load_iris()

iris_classifier = LogisticRegressionCV()
iris_classifier.fit(iris.data, iris.target)

sklearn_joblib.dump(iris_classifier, "dump/" + _platform_module("sklearn_joblib", sklearn_joblib.__version__) + ".pkl.z", compress = True)
joblib.dump(iris_classifier, "dump/" + _platform_module("joblib", joblib.__version__) + ".pkl.z", compress = True)

for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
	_pickle(protocol)
