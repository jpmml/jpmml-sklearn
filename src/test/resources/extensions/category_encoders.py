from common import *

from sklearn.experimental import enable_hist_gradient_boosting

from category_encoders import BaseNEncoder, BinaryEncoder, CatBoostEncoder, CountEncoder, LeaveOneOutEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder, WOEEncoder
from mlxtend.preprocessing import DenseTransformer
from pandas import DataFrame
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder as SkLearnOneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

import numpy

cat_cols = ["Employment", "Education", "Marital", "Occupation", "Gender"]
cont_cols = ["Age", "Income", "Hours"]

def load_filter_audit(name, filter = False):
	if name.endswith("Audit"):
		audit_X, audit_y = load_audit("Audit")
	elif name.endswith("AuditNA"):
		audit_X, audit_y = load_audit("AuditNA")
	else:
		raise ValueError()

	if filter:
		mask = numpy.ones((audit_X.shape[0], ), dtype = bool)

		mask = numpy.logical_and(mask, audit_X.Employment != "SelfEmp")
		mask = numpy.logical_and(mask, audit_X.Education != "Professional")
		mask = numpy.logical_and(mask, audit_X.Marital != "Divorced")
		mask = numpy.logical_and(mask, audit_X.Occupation != "Service")

		audit_X = audit_X[mask]
		audit_y = audit_y[mask]

	return (audit_X, audit_y)

def can_handle_unknown(cat_encoder):
	if isinstance(cat_encoder, Pipeline):
		cat_encoder = cat_encoder.steps[0][1]

	if isinstance(cat_encoder, OrdinalEncoder):
		return (cat_encoder.handle_unknown != "error") and (cat_encoder.handle_unknown != "value")
	else:
		return cat_encoder.handle_unknown != "error"

def build_audit(cat_encoder, cont_encoder, classifier, name, **pmml_options):
	audit_X, audit_y = load_filter_audit(name, can_handle_unknown(cat_encoder))

	steps = [
		("mapper", ColumnTransformer([
			("cat", cat_encoder, cat_cols),
			("cont", cont_encoder, cont_cols)
		])),
		("classifier", classifier)
	]

	if isinstance(classifier, HistGradientBoostingClassifier):
		steps.insert(1, ("formatter", DenseTransformer()))

	pipeline = PMMLPipeline(steps)
	pipeline.fit(audit_X, audit_y)
	pipeline.configure(**pmml_options)
	store_pkl(pipeline, name)

	audit_X, audit_y = load_filter_audit(name, False)

	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

classifier = LogisticRegression()

build_audit(OneHotEncoder(handle_missing = "error", handle_unknown = "error"), "passthrough", clone(classifier), "OneHotEncoderAudit")
build_audit(Pipeline([("ordinal", OrdinalEncoder(handle_missing = "error", handle_unknown = "value")), ("ohe", SkLearnOneHotEncoder())]), "passthrough", clone(classifier), "OrdinalEncoderAudit")

classifier = HistGradientBoostingClassifier(random_state = 13)

build_audit(OneHotEncoder(handle_missing = "value", handle_unknown = "error"), "passthrough", clone(classifier), "OneHotEncoderAuditNA")
build_audit(Pipeline([("ordinal", OrdinalEncoder(handle_missing = "value", handle_unknown = "error")), ("ohe", SkLearnOneHotEncoder())]), "passthrough", clone(classifier), "OrdinalEncoderAuditNA")

classifier = LogisticRegression()

build_audit(BaseNEncoder(base = 2, drop_invariant = True, handle_missing = "error", handle_unknown = "error"), "passthrough", clone(classifier), "Base2EncoderAudit")
build_audit(BaseNEncoder(base = 2, handle_missing = "value", handle_unknown = "error"), SimpleImputer(), clone(classifier), "Base2EncoderAuditNA")
build_audit(Pipeline([("basen", BaseNEncoder(base = 3, drop_invariant = True, handle_missing = "error", handle_unknown = "error")), ("ohe", SkLearnOneHotEncoder())]), "passthrough", clone(classifier), "Base3EncoderAudit")
build_audit(Pipeline([("basen", BaseNEncoder(base = 3, handle_missing = "value", handle_unknown = "error")), ("ohe", SkLearnOneHotEncoder())]), SimpleImputer(), clone(classifier), "Base3EncoderAuditNA")

xgb_pmml_options = {"compact": False, "numeric": False}

classifier = XGBClassifier(n_estimators = 101, random_state = 13)

build_audit(BaseNEncoder(base = 4, drop_invariant = True, handle_missing = "error", handle_unknown = "error"), "passthrough", clone(classifier), "Base4EncoderAudit", **xgb_pmml_options)
build_audit(BaseNEncoder(base = 4, handle_missing = "value", handle_unknown = "error"), "passthrough", clone(classifier), "Base4EncoderAuditNA", **xgb_pmml_options)

rf_pmml_options = {"compact" : False, "numeric": False}

classifier = RandomForestClassifier(n_estimators = 71, random_state = 13)

build_audit(BinaryEncoder(handle_missing = "error", handle_unknown = "error"), "passthrough", clone(classifier), "BinaryEncoderAudit", **rf_pmml_options)

rf_pmml_options = {"compact" : False, "numeric": True}

build_audit(CatBoostEncoder(a = 0.5, handle_missing = "error", handle_unknown = "value"), "passthrough", clone(classifier), "CatBoostEncoderAudit", **rf_pmml_options)
build_audit(CountEncoder(normalize = True, min_group_size = 0.05, handle_missing = "error", handle_unknown = "value"), "passthrough", clone(classifier), "CountEncoderAudit", **rf_pmml_options)
build_audit(LeaveOneOutEncoder(handle_missing = "error", handle_unknown = "value"), "passthrough", clone(classifier), "LeaveOneOutEncoderAudit", **rf_pmml_options)

build_audit(TargetEncoder(handle_missing = "error", handle_unknown = "value"), "passthrough", clone(classifier), "TargetEncoderAudit", **rf_pmml_options)
build_audit(WOEEncoder(handle_missing = "error", handle_unknown = "value"), "passthrough", clone(classifier), "WOEEncoderAudit", **rf_pmml_options)

classifier = XGBClassifier(n_estimators = 101, random_state = 13)

build_audit(BinaryEncoder(handle_missing = "value", handle_unknown = "error"), "passthrough", clone(classifier), "BinaryEncoderAuditNA", **xgb_pmml_options)

xgb_pmml_options = {"compact": False, "numeric": True, "prune": True}

build_audit(CatBoostEncoder(a = 0.5, handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "CatBoostEncoderAuditNA", **xgb_pmml_options)
build_audit(CountEncoder(min_group_size = 10, handle_missing = "value", handle_unknown = 10), "passthrough", clone(classifier), "CountEncoderAuditNA", **xgb_pmml_options)
build_audit(LeaveOneOutEncoder(handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "LeaveOneOutEncoderAuditNA", **xgb_pmml_options)

xgb_pmml_options = {"compact": False, "numeric": True}

build_audit(TargetEncoder(handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "TargetEncoderAuditNA", **xgb_pmml_options)
build_audit(WOEEncoder(handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "WOEEncoderAuditNA", **xgb_pmml_options)