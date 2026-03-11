import sys

sys.path.append("../../../../pmml-sklearn-extension/src/test/resources/")

from extensions.causalml import *

from causalml.inference.meta import XGBRRegressor, XGBTRegressor

email_df = load_csv("Email")

def _set_control_name(estimator, control_name):
	estimator.control_name = control_name
	return estimator

build_email_parallel(email_df, _set_control_name(XGBRRegressor(effect_learner_n_estimators = 31, max_depth = 5, random_state = 42), control_name = "control"), "XGBRRegressorEmail")
build_email_parallel(email_df, XGBTRegressor(n_estimators = 31, max_depth = 5, random_state = 42, control_name = "control"), "XGBTRegressorEmail")
