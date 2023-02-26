import sys

from lightgbm import LGBMClassifier, LGBMRegressor

sys.path.append("../../../../pmml-sklearn/src/test/resources/")

from main import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Iris"]

if "Audit" in datasets:
	audit_df = load_audit("Audit", stringify = False)

	build_audit(audit_df, LGBMClassifier(objective = "binary", n_estimators = 37), "LGBMAudit", predict_params = {"num_iteration" : 17}, predict_proba_params = {"num_iteration" : 17}, num_iteration = 17)

def build_audit_cat(audit_df, classifier, name, with_proba = True, fit_params = {}):
	audit_X, audit_y = split_csv(audit_df)

	marital_mapping = {
		"Married-spouse-absent" : "Married"
	}
	mapper = DataFrameMapper(
		[(["Age"], [ContinuousDomain(display_name = "Age"), KBinsDiscretizer(n_bins = 7, encode = "ordinal", strategy = "quantile")])] +
		[(["Income"], [ContinuousDomain(display_name = "Income"), KBinsDiscretizer(n_bins = 11, encode = "ordinal", strategy = "kmeans")])] +
		[(["Hours"], [ContinuousDomain(display_name = "Hours"), CutTransformer(bins = [0, 20, 40, 60, 80, 100], labels = False, right = False, include_lowest = True)])] +
		[(["Employment", "Education"], [MultiDomain([CategoricalDomain(display_name = "Employment"), CategoricalDomain(display_name = "Education")]), OrdinalEncoder(dtype = numpy.int_)])] +
		[(["Marital"], [CategoricalDomain(display_name = "Marital"), FilterLookupTransformer(marital_mapping), OrdinalEncoder(dtype = numpy.uint16)])] +
		[(["Occupation"], [CategoricalDomain(display_name = "Occupation"), OrdinalEncoder(dtype = numpy.float_)])] +
		[([column], [CategoricalDomain(display_name = column), LabelEncoder()]) for column in ["Gender", "Deductions"]]
	)
	pipeline = Pipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y, **fit_params)
	pipeline = make_pmml_pipeline(pipeline, audit_X.columns.values, audit_y.name)
	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	if with_proba:
		adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

if "Audit" in datasets:
	audit_df = load_audit("Audit")

	cat_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

	build_audit_cat(audit_df, GBDTLRClassifier(LGBMClassifier(n_estimators = 17, random_state = 13), LogisticRegression()), "LGBMLRAuditCat", fit_params = {"classifier__gbdt__categorical_feature" : cat_indices})
	build_audit_cat(audit_df, LGBMClassifier(objective = "binary", n_estimators = 101), "LGBMAuditCat", fit_params = {"classifier__categorical_feature" : cat_indices})

if "Iris" in datasets:
	iris_df = load_iris("Iris")
	iris_X, iris_y = split_csv(iris_df)

	build_iris_opt(iris_df, LGBMClassifier(objective = "multiclass"), "LGBMIris", fit_params = {"classifier__eval_set" : [(iris_X[iris_test_mask], iris_y[iris_test_mask])], "classifier__eval_metric" : "multi_logloss", "classifier__early_stopping_rounds" : 3})

if "Auto" in datasets:
	auto_df = load_auto("Auto")
	auto_X, auto_y = split_csv(auto_df)

	build_auto_opt(auto_df, LGBMRegressor(objective = "regression", random_state = 13), "LGBMAuto", fit_params = {"regressor__eval_set" : [(auto_X[auto_test_mask], auto_y[auto_test_mask])], "regressor__eval_metric" : "rmse", "regressor__early_stopping_rounds" : 3})
