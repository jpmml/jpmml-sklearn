from xgboost import XGBClassifier

import os
import sys

sys.path.append(os.path.abspath("../../../../pmml-sklearn/src/test/resources/"))

sys.path.append(os.path.abspath("../../../../pmml-sklearn-extension/src/test/resources/"))
sys.path.append(os.path.abspath("../../../../pmml-sklearn-extension/src/test/resources/extensions/"))

from extensions.category_encoders import *

xgb_pmml_options = {"compact": False, "numeric": False}

classifier = XGBClassifier(n_estimators = 101, random_state = 13)

build_audit(BaseNEncoder(base = 4, drop_invariant = True, handle_missing = "error", handle_unknown = "error"), "passthrough", clone(classifier), "Base4EncoderAudit", **xgb_pmml_options)

xgb_pmml_options = {"compact": False, "numeric": True}

build_audit(BaseNEncoder(base = 4, handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "Base4EncoderAuditNA", **xgb_pmml_options)

xgb_pmml_options = {"compact": False, "numeric": True, "prune": True}

classifier = XGBClassifier(n_estimators = 101, random_state = 13)

build_audit(BinaryEncoder(handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "BinaryEncoderAuditNA", **xgb_pmml_options)
build_audit(CatBoostEncoder(a = 0.5, handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "CatBoostEncoderAuditNA", **xgb_pmml_options)
build_audit(CountEncoder(min_group_size = 10, handle_missing = "value", handle_unknown = 10), "passthrough", clone(classifier), "CountEncoderAuditNA", **xgb_pmml_options)
build_audit(LeaveOneOutEncoder(handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "LeaveOneOutEncoderAuditNA", **xgb_pmml_options)

xgb_pmml_options = {"compact": False, "numeric": True}

build_audit(TargetEncoder(handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "TargetEncoderAuditNA", **xgb_pmml_options)
build_audit(WOEEncoder(handle_missing = "value", handle_unknown = "value"), "passthrough", clone(classifier), "WOEEncoderAuditNA", **xgb_pmml_options)