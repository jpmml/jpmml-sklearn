from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer, PowerTransformer, QuantileTransformer, SplineTransformer
from sklearn2pmml.pipeline import PMMLPipeline

import numpy

from common import *

def make_dummy_pipeline(transformer, n_outputs = 1):
	regressor = LinearRegression()
	if n_outputs == 1:
		regressor.coef_ = numpy.ones(transformer.n_features_in_)
		regressor.intercept_ = numpy.zeros(1)
	else:
		regressor.coef_ = numpy.eye(n_outputs)
		regressor.intercept_ = numpy.zeros(n_outputs)

	pipeline = PMMLPipeline([
		("regressor", regressor),
	], predict_transformer = transformer)

	return pipeline

n_samples = 27

def build_spline(prefix, degree, n_knots):
	name = prefix + "Spline"

	transformer = SplineTransformer(extrapolation = "error", degree = degree, n_knots = n_knots)
	transformer.fit(train)

	pipeline = make_dummy_pipeline(transformer)
	store_pkl(pipeline, name)

	y = DataFrame(pipeline.predict_transform(test), columns = ["y"] + ["bspline(predict(y), {})".format(i) for i in range(0, transformer.n_features_out_)])
	y.to_csv("csv/" + name + ".csv", index = False, sep = "\t")

train = numpy.linspace(0, 1, n_samples).reshape(-1, 1)
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

train = numpy.linspace(0.1, 0.9, n_samples).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])
store_csv(train, "BoxCox")

test = load_csv("BoxCox")

build_power("PlainBoxCox", method = "box-cox", standardize = False)
build_power("StandardizedBoxCox", method = "box-cox", standardize = True)

train = numpy.linspace(-1, 1, n_samples).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])
store_csv(train, "YeoJohnson")

test = load_csv("YeoJohnson")

build_power("PlainYeoJohnson", method = "yeo-johnson", standardize = False)
build_power("StandardizedYeoJohnson", method = "yeo-johnson", standardize = True)

n_samples = 1000

def build_norm(name, norm):
	transformer = Normalizer(norm = norm)
	transformer.fit(train)

	pipeline = make_dummy_pipeline(transformer, n_outputs = len(train.columns))
	store_pkl(pipeline, name)

	columns = ["y{}".format(i + 1) for i in range(0, transformer.n_features_in_)] + ["normalizer(predict(y{}))".format(i + 1) for i in range(0, transformer.n_features_in_)]

	y = DataFrame(pipeline.predict_transform(test), columns = columns)
	y.to_csv("csv/" + name + ".csv", index = False, sep = "\t")

train = numpy.column_stack([
	numpy.random.uniform(-10, 10, n_samples),
	numpy.random.uniform(-100, 100, n_samples),
	numpy.random.exponential(5, n_samples),
	numpy.random.normal(0, 20, n_samples),
	numpy.random.uniform(-1, 1, n_samples)
])
train = DataFrame(train, columns = ["x{}".format(i + 1) for i in range(0, 5)])
train.to_csv("csv/Norm.csv", index = False, sep = "\t")

test = pandas.read_csv("csv/Norm.csv", na_values = ["N/A", "NA"], sep = "\t")

build_norm("L1Norm", norm = "l1")
build_norm("L2Norm", norm = "l2")
build_norm("MaxNorm", norm = "max")

def build_quantile(name, output_distribution):
	transformer = QuantileTransformer(output_distribution = output_distribution, n_quantiles = 100, subsample = None)
	transformer.fit(train)

	pipeline = make_dummy_pipeline(transformer)
	store_pkl(pipeline, name)

	y = DataFrame(pipeline.predict_transform(test), columns = ["y", "quantile(predict(y))"])
	y.to_csv("csv/" + name + ".csv", index = False, sep = "\t")

train = numpy.random.lognormal(mean = 0, sigma = 1, size = n_samples).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])
store_csv(train, "Quantile")

test = load_csv("Quantile")

build_quantile("UniformQuantile", output_distribution = "uniform")
build_quantile("NormalQuantile", output_distribution = "normal")
