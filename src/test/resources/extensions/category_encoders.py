from common import *

from category_encoders import BaseNEncoder, BinaryEncoder, OrdinalEncoder
from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

audit_X, audit_y = load_audit("Audit")

cat_cols = ["Employment", "Education", "Marital", "Occupation", "Gender"]
cont_cols = ["Age", "Income", "Hours"]

def build_audit(mapper, classifier, name, **pmml_options):
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	pipeline.configure(**pmml_options)
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
	(cat_cols, [BaseNEncoder(base = 3, drop_invariant = True), OneHotEncoder()]),
	(cont_cols, None)
])

build_audit(mapper, classifier, "Base3EncoderAudit")

classifier = XGBClassifier(objective = "binary:logistic", n_estimators = 31, max_depth = 7, random_state = 13)

mapper = DataFrameMapper([
	(cat_cols, BaseNEncoder(base = 4, drop_invariant = True)),
	(cont_cols, None)
])

build_audit(mapper, classifier, "Base4EncoderAudit", compact = False)

classifier = RandomForestClassifier(n_estimators = 31, random_state = 13)

mapper = DataFrameMapper([
	(cat_cols, BinaryEncoder()),
	(cont_cols, None)
])

build_audit(mapper, classifier, "BinaryEncoderAudit", compact = False)