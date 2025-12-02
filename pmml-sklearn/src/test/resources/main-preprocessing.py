from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, SplineTransformer
from sklearn2pmml.pipeline import PMMLPipeline

import numpy

from common import *

def make_dummy_pipeline(transformer):
	regressor = LinearRegression()
	regressor.coef_ = numpy.array([1])
	regressor.intercept_ = numpy.array([0])

	pipeline = PMMLPipeline([
		("regressor", regressor),
	], predict_transformer = transformer)

	return pipeline

def build_spline(prefix, degree, n_knots):
	name = prefix + "Spline"

	transformer = SplineTransformer(extrapolation = "error", degree = degree, n_knots = n_knots)
	transformer.fit(train)

	pipeline = make_dummy_pipeline(transformer)
	store_pkl(pipeline, name)

	y = DataFrame(pipeline.predict_transform(test), columns = ["y"] + ["bspline(predict(y), {})".format(i) for i in range(0, transformer.n_features_out_)])
	y.to_csv("csv/" + name + ".csv", index = False, sep = "\t")

train = numpy.linspace(0, 1, 27).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])
store_csv(train, "Spline")

test = load_csv("Spline")

build_spline("Quadratic2", degree = 2, n_knots = 2)
build_spline("Quadratic3", degree = 2, n_knots = 3)
build_spline("Quadratic4", degree = 2, n_knots = 4)
build_spline("Quadratic5", degree = 2, n_knots = 5)

build_spline("Cubic2", degree = 3, n_knots = 2)
build_spline("Cubic3", degree = 3, n_knots = 3)

def build_power(name, method, standardize):
	transformer = PowerTransformer(method = method, standardize = standardize)
	transformer.fit(train)

	pipeline = make_dummy_pipeline(transformer)
	store_pkl(pipeline, name)

	y = DataFrame(pipeline.predict_transform(test), columns = ["y", "standardScaler(power(predict(y)))" if standardize else "power(predict(y))"])
	y.to_csv("csv/" + name + ".csv", index = False, sep = "\t")

train = numpy.linspace(0.1, 0.9, 27).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])
store_csv(train, "BoxCox")

test = load_csv("BoxCox")

build_power("PlainBoxCox", method = "box-cox", standardize = False)
build_power("StandardizedBoxCox", method = "box-cox", standardize = True)

train = numpy.linspace(-1, 1, 27).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])
store_csv(train, "YeoJohnson")

test = load_csv("YeoJohnson")

build_power("PlainYeoJohnson", method = "yeo-johnson", standardize = False)
build_power("StandardizedYeoJohnson", method = "yeo-johnson", standardize = True)

def build_quantile(name, output_distribution):
	transformer = QuantileTransformer(output_distribution = output_distribution, n_quantiles = 100, subsample = None)
	transformer.fit(train)

	pipeline = make_dummy_pipeline(transformer)
	store_pkl(pipeline, name)

	y = DataFrame(pipeline.predict_transform(test), columns = ["y", "quantile(predict(y))"])
	y.to_csv("csv/" + name + ".csv", index = False, sep = "\t")

train = numpy.random.lognormal(mean = 0, sigma = 1, size = 1000).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])
store_csv(train, "Quantile")

test = load_csv("Quantile")

build_quantile("UniformQuantile", output_distribution = "uniform")
build_quantile("NormalQuantile", output_distribution = "normal")
