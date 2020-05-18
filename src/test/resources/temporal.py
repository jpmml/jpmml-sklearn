from common import *

from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.decoration import DateTimeDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import ExpressionTransformer, DaysSinceYearTransformer, SecondsSinceYearTransformer

df = DataFrame([
	["1968-12-21T12:51:00", None, "1968-12-27T15:51:42", True], # Apollo 8
	["1969-05-18T16:49:00", None, "1969-05-26T16:52:23", True], # Apollo 10
	["1969-07-16T13:32:00", "1969-07-20T20:17:40", "1969-07-24T16:50:35", True], # Apollo 11
	["1969-11-14T16:22:00", "1969-11-19T06:54:35", "1969-11-24T20:58:24", True], # Apollo 12
	["1970-04-11T19:13:00", None, "1970-04-17T18:07:41", False], # Apollo 13
	["1971-01-31T21:03:02", "1971-02-05T09:18:11", "1971-02-09T21:05:00", True], # Apollo 14
	["1971-07-26T13:34:00", "1971-07-30T22:16:29", "1971-08-07T20:45:53", True], # Apollo 15
	["1972-04-16T17:54:00", "1972-04-21T02:23:35", "1972-04-27T19:45:05", True], # Apollo 16
	["1972-12-07T05:33:00", "1972-12-11T19:54:57", "1972-12-19T19:24:59", True], # Apollo 17
], columns = ["launch", "moon landing", "return", "success"])

store_csv(df, "Apollo")

def build_apollo(mapper, name):
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", DecisionTreeClassifier())
	])
	pipeline.fit(df, df["success"])
	store_pkl(pipeline, name)
	success = DataFrame(pipeline.predict(df), columns = ["success"])
	success_proba = DataFrame(pipeline.predict_proba(df), columns = ["probability(false)", "probability(true)"])
	success = pandas.concat((success, success_proba), axis = 1)
	store_csv(success, name)

mapper = DataFrameMapper([
	(["launch", "return"], [DateTimeDomain(), DaysSinceYearTransformer(year = 1968), ExpressionTransformer("X[1] - X[0]")])
])

build_apollo(mapper, "DurationInDaysApollo")

mapper = DataFrameMapper([
	(["launch", "return"], [DateTimeDomain(), SecondsSinceYearTransformer(year = 1968), ExpressionTransformer("X[1] - X[0]")])
])

build_apollo(mapper, "DurationInSecondsApollo")