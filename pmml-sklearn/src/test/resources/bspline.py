from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn2pmml.pipeline import PMMLPipeline

import numpy

from common import *

train = numpy.linspace(0, 1, 13).reshape(-1, 1)
train = DataFrame(train, columns = ["x1"])

test = load_csv("BSpline")

def build_bspline(prefix, **transformer_kwargs):
	name = prefix + "BSpline"

	regressor = LinearRegression()
	regressor.coef_ = numpy.array([1])
	regressor.intercept_ = numpy.array([0])

	transformer = SplineTransformer(extrapolation = "error", **transformer_kwargs)
	transformer.fit(train)

	pipeline = PMMLPipeline([
		("regressor", regressor)
	], predict_transformer = transformer)
	store_pkl(pipeline, name)

	y = DataFrame(pipeline.predict_transform(test), columns = ["y"] + ["bspline(predict(y), {})".format(i) for i in range(0, transformer.n_features_out_)])
	y.to_csv("csv/" + name + ".csv", index = False, sep = "\t")

build_bspline("Quadratic2", degree = 2, n_knots = 2)
build_bspline("Quadratic3", degree = 2, n_knots = 3)
build_bspline("Quadratic4", degree = 2, n_knots = 4)
build_bspline("Quadratic5", degree = 2, n_knots = 5)

build_bspline("Cubic2", degree = 3, n_knots = 2)
build_bspline("Cubic3", degree = 3, n_knots = 3)
