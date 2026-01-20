import sys

sys.path.append("../../../../pmml-sklearn-extension/src/test/resources/")

from extensions.flaml import *

audit_df = load_audit("Audit")

build_audit(audit_df, "xgb_limitdepth", "FLAMLLimitDepthAudit", binarize = False, standardize = False)
build_audit(audit_df, "xgboost", "FLAMLAudit", binarize = False, standardize = False)

auto_df = load_auto("Auto")

build_auto(auto_df, "xgb_limitdepth", "FLAMLLimitDepthAuto", binarize = False, standardize = False)
build_auto(auto_df, "xgboost", "FLAMLAuto", binarize = False, standardize = False)
