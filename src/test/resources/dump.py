from sklearn.datasets import load_iris
from sklearn.externals import joblib as sklearn_joblib
from sklearn.linear_model import LogisticRegressionCV

import joblib
import numpy
import pickle
import platform

def _platform():
	version = platform.python_version_tuple()
	return ("python-" + version[0] + "." + version[1])

def _module(name, version):
	return (name + "-" + version)

def _platform_module(name, version):
	return (_platform() + "_" + _module(name, version))

def _pickle(obj, path):
	con = open(path, "wb")
	pickle.dump(obj, con, protocol = protocol)
	con.close()

iris = load_iris()

iris_classifier = LogisticRegressionCV()
iris_classifier.fit(iris.data, iris.target)

sklearn_joblib.dump(iris_classifier, "dump/" + _platform_module("sklearn-joblib", sklearn_joblib.__version__) + ".pkl.z", compress = True)
joblib.dump(iris_classifier, "dump/" + _platform_module("joblib", joblib.__version__) + ".pkl.z", compress = True)

for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
	_pickle(iris_classifier, "dump/" + _platform_module("pickle", "p" + str(protocol)) + ".pkl")

def _pickle_values(values, dtype):
	_pickle(values.astype(dtype), "dump/" + _platform_module("numpy", numpy.__version__) + "_" + dtype.__name__ + ".pkl")

values = numpy.asarray([0, 1], dtype = numpy.int8)
_pickle_values(values, bool)

values = numpy.asarray([x for x in range(-128, 127, 1)], dtype = numpy.int8)
_pickle_values(values, numpy.int8)

values = numpy.asarray([x for x in range(0, 255, 1)], dtype = numpy.uint8)
_pickle_values(values, numpy.uint8)

values = numpy.asarray([x for x in range(-32768, 32767, 127)], dtype = numpy.int16)
_pickle_values(values, numpy.int16)

values = numpy.asarray([x for x in range(0, 65535, 127)], dtype = numpy.uint16)
_pickle_values(values, numpy.uint16)

values = numpy.asarray([x for x in range(-2147483648, 2147483647, 64 * 32767)], dtype = numpy.int32)
_pickle_values(values, int)
_pickle_values(values, numpy.int32)
_pickle_values(values, numpy.int64)
_pickle_values(values, numpy.float32)
_pickle_values(values, numpy.float64)

values = numpy.asarray([x for x in range(0, 4294967295, 64 * 32767)], dtype = numpy.uint32)
_pickle_values(values, numpy.uint32)
_pickle_values(values, numpy.uint64)