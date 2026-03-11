import sys

sys.path.append("../../../../pmml-sklearn-extension/src/test/resources/")

from extensions.causalml import *

from causalml.inference.meta import XGBTRegressor

email_df = load_csv("Email")

build_email_parallel(email_df, XGBTRegressor(n_estimators = 31, max_depth = 5, random_state = 42, control_name = "control"), "XGBTRegressorEmail")
