import sys

sys.path.append("../../../../pmml-sklearn-extension/src/test/resources/")

from extensions.flaml import *

audit_df = load_audit("Audit")

build_audit(audit_df, "lgbm", "FLAMLAudit", binarize = False, standardize = False)

auto_df = load_auto("Auto")

build_auto(auto_df, "lgbm", "FLAMLAuto", binarize = False, standardize = False)
