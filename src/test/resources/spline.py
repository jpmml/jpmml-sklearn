from common import *

from pandas import DataFrame
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.scipy import BSplineTransformer

import numpy

# See https://ndsplines.readthedocs.io/en/latest/auto_examples/1d-interp.html

def gaussian(x):
	z = norm.ppf(0.9995)
	x = z * (2 * x - 1)
	return norm.pdf(x)

def sin(x):
	x = numpy.pi*(x - 0.5)
	return numpy.sin(x)

def tanh(x):
	x = 2 * numpy.pi * (x - 0.5)
	return numpy.tanh(x)

train = numpy.linspace(0, 1, 9)
test = numpy.linspace(0.01, 0.99, 12)

X = DataFrame(test.reshape(-1, 1), columns = ["x1"])
store_csv(X, "BSpline")

for fun in [gaussian, sin, tanh]:
	name = fun.__name__.capitalize() + "BSpline"

	bspline = make_interp_spline(train, fun(train), k = 3)

	regressor = LinearRegression()
	regressor.coef_ = numpy.array([1])
	regressor.intercept_ = numpy.array([0])

	pipeline = PMMLPipeline([
		("regressor", regressor)
	], predict_transformer = BSplineTransformer(bspline))
	store_pkl(pipeline, name)

	y = DataFrame(pipeline.predict_transform(X), columns = ["y", "bspline(predict(y))"])
	store_csv(y, name)