from common import *

from category_encoders import BaseNEncoder, BinaryEncoder, CountEncoder, OrdinalEncoder, TargetEncoder, WOEEncoder
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

audit_X, audit_y = load_audit("Audit")

cat_cols = ["Employment", "Education", "Marital", "Occupation", "Gender"]
cont_cols = ["Age", "Income", "Hours"]

def build_audit(transformer, classifier, name, **pmml_options):
	pipeline = PMMLPipeline([
		("transformer", transformer),
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

transformer = ColumnTransformer([
	("cat", Pipeline([("ordinal", OrdinalEncoder()), ("ohe", OneHotEncoder())]), cat_cols),
	("cont", "passthrough", cont_cols)
])

build_audit(transformer, classifier, "OrdinalEncoderAudit")

transformer = ColumnTransformer([
	("cat", BaseNEncoder(base = 2, drop_invariant = True), cat_cols),
	("cont", "passthrough", cont_cols)
])

build_audit(transformer, classifier, "Base2EncoderAudit")

transformer = ColumnTransformer([
	("cat", Pipeline([("basen", BaseNEncoder(base = 3, drop_invariant = True)), ("ohe", OneHotEncoder())]), cat_cols),
	("cont", "passthrough", cont_cols)
])

build_audit(transformer, classifier, "Base3EncoderAudit")

classifier = XGBClassifier(objective = "binary:logistic", n_estimators = 31, max_depth = 7, random_state = 13)

transformer = ColumnTransformer([
	("cat", BaseNEncoder(base = 4, drop_invariant = True), cat_cols),
	("cont", "passthrough", cont_cols)
])

build_audit(transformer, classifier, "Base4EncoderAudit", compact = False)

classifier = RandomForestClassifier(n_estimators = 31, random_state = 13)

transformer = ColumnTransformer([
	("cat", BinaryEncoder(), cat_cols),
	("cont", "passthrough", cont_cols)
])

build_audit(transformer, classifier, "BinaryEncoderAudit", compact = False)

transformer = ColumnTransformer([
	("cat_int", CountEncoder(min_group_size = 100), ["Education", "Employment", "Occupation"]),
	("cat_float64", CountEncoder(normalize = True, min_group_size = 0.05), ["Marital", "Gender"]),
	("cont", "passthrough", cont_cols)
])

classifier = RandomForestClassifier(n_estimators = 31, random_state = 13)

build_audit(transformer, classifier, "CountEncoderAudit", compact = False)

transformer = ColumnTransformer([
	("cat", TargetEncoder(), cat_cols),
	("cont", "passthrough", cont_cols)
])

classifier = RandomForestClassifier(n_estimators = 31, random_state = 13)

build_audit(transformer, classifier, "TargetEncoderAudit", compact = False)

transformer = ColumnTransformer([
	("cat", WOEEncoder(), cat_cols),
	("cont", "passthrough", cont_cols)
])

classifier = RandomForestClassifier(n_estimators = 31, random_state = 13)

build_audit(transformer, classifier, "WOEEncoderAudit", compact = False)