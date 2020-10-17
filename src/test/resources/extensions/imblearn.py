from common import *

from imblearn.ensemble import BalancedBaggingClassifier
from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import ExpressionTransformer

audit_X, audit_y = load_audit("Audit")

def build_audit(classifier, name):
	mapper = DataFrameMapper(
		[(cat_column, [CategoricalDomain(), LabelBinarizer()]) for cat_column in ["Employment", "Education", "Marital", "Occupation", "Gender"]] +
		[(cont_column, ContinuousDomain()) for cont_column in ["Age", "Income", "Hours"]] +
		[(["Income", "Hours"], Alias(ExpressionTransformer("X[0] / (X[1] * 52.0)"), "Hourly_Income", prefit = True))]
	, df_out = True)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	pipeline.verify(audit_X.sample(frac = 0.05, random_state = 13))
	store_pkl(pipeline, name)
	adjusted = DataFrame(pipeline.predict(audit_X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(audit_X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name)

build_audit(BalancedBaggingClassifier(sampling_strategy = "auto", n_estimators = 3, random_state = 13), "BalancedDecisionTreeEnsembleAudit")