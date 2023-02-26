import sys

from xgboost.sklearn import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from main import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Housing", "Iris", "Sentiment", "Versicolor"]

if "Audit" in datasets:
	audit_df = load_audit("Audit", stringify = False)

	build_audit(audit_df, GBDTLRClassifier(XGBClassifier(n_estimators = 17, random_state = 13), LogisticRegression()), "XGBLRAudit")
	build_audit(audit_df, GBDTLRClassifier(XGBRFClassifier(n_estimators = 7, max_depth = 6, random_state = 13), SGDClassifier(loss = "log", penalty = "elasticnet", random_state = 13)), "XGBRFLRAudit")
	build_audit(audit_df, XGBClassifier(objective = "binary:logistic", importance_type = "weight", random_state = 13), "XGBAudit", predict_params = {"ntree_limit" : 71}, predict_proba_params = {"ntree_limit" : 71}, byte_order = "LITTLE_ENDIAN", charset = "US-ASCII", ntree_limit = 71)
	build_audit(audit_df, XGBRFClassifier(objective = "binary:logistic", n_estimators = 31, max_depth = 5, random_state = 13), "XGBRFAudit")

def build_audit_na_direct(audit_na_df, classifier, name):
	audit_na_X, audit_na_y = split_csv(audit_na_df)

	mapper = DataFrameMapper([
		(["Age", "Hours", "Income"], None),
		(["Education", "Employment"], [SimpleImputer(strategy = "constant", fill_value = "Unknown"), OrdinalEncoder(), OneHotEncoder()]),
		(["Gender"], [SimpleImputer(strategy = "constant", fill_value = "Unknown"), OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = 11), OneHotEncoder()]),
		(["Marital", "Occupation"], OneHotEncoder())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_na_X, audit_na_y)
	pipeline.verify(audit_na_X.sample(frac = 0.05, random_state = 13), precision = 1e-5, zeroThreshold = 1e-5)
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_na_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_na_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_na_df = load_audit("AuditNA")

	build_audit_na_direct(audit_na_df, XGBClassifier(objective = "binary:logistic", use_label_encoder = False, random_state = 13), "XGBAuditNA")

if "Versicolor" in datasets:
	versicolor_df = load_versicolor("Versicolor")

	build_versicolor(versicolor_df, GBDTLRClassifier(XGBRFClassifier(n_estimators = 7, random_state = 13), LinearSVC(random_state = 13)), "XGBRFLRVersicolor", with_proba = False)

if "Iris" in datasets:
	iris_df = load_iris("Iris")
	iris_X, iris_y = split_csv(iris_df)

	build_iris_opt(iris_df, XGBClassifier(objective = "multi:softprob"), "XGBIris", fit_params = {"classifier__eval_set" : [(iris_X[iris_test_mask], iris_y[iris_test_mask])], "classifier__eval_metric" : "mlogloss", "classifier__early_stopping_rounds" : 3})

if "Auto" in datasets:
	auto_df = load_auto("Auto")

	auto_df["cylinders"] = auto_df["cylinders"].astype(int)
	auto_df["model_year"] = auto_df["model_year"].astype(int)
	auto_df["origin"] = auto_df["origin"].astype(int)

	build_auto(auto_df, GBDTLMRegressor(XGBRFRegressor(n_estimators = 17, max_depth = 6, random_state = 13), ElasticNet(random_state = 13)), "XGBRFLMAuto")
	build_auto(auto_df, XGBRFRegressor(n_estimators = 31, max_depth = 6, random_state = 13), "XGBRFAuto")

if "Auto" in datasets:
	auto_X, auto_y = split_csv(auto_df)

	build_auto_opt(auto_df, XGBRegressor(objective = "reg:squarederror", missing = 1, random_state = 13), "XGBAuto", fit_params = {"regressor__eval_set" : [(auto_X[auto_test_mask], auto_y[auto_test_mask])], "regressor__eval_metric" : "rmse", "regressor__early_stopping_rounds" : 3})

if "Housing" in datasets:
	housing_df = load_housing("Housing")

	build_housing(housing_df, GBDTLMRegressor(XGBRFRegressor(n_estimators = 17, max_depth = 5, random_state = 13), SGDRegressor(penalty = "elasticnet", random_state = 13)), "XGBRFLMHousing")
