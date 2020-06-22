from common import *

from category_encoders import BaseNEncoder, OrdinalEncoder
from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline

audit_X, audit_y = load_audit("Audit")

cat_cols = ["Employment", "Education", "Marital", "Occupation", "Gender"]
cont_cols = ["Age", "Income", "Hours"]

def build_audit(mapper, classifier, name):
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

classifier = LogisticRegression(multi_class = "multinomial")

mapper = DataFrameMapper([
	(cat_cols, [OrdinalEncoder(), OneHotEncoder()]),
	(cont_cols, None)
])

build_audit(mapper, classifier, "OrdinalEncoderAudit")

mapper = DataFrameMapper([
	(cat_cols, BaseNEncoder(base = 2, drop_invariant = True)),
	(cont_cols, None)
])

build_audit(mapper, classifier, "Base2EncoderAudit")

mapper = DataFrameMapper([
	(cat_cols, BaseNEncoder(base = 3, drop_invariant = True)),
	(cont_cols, None)
])

build_audit(mapper, classifier, "Base3EncoderAudit")