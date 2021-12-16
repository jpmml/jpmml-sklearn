/*
 * Copyright (c) 2015 Villu Ruusmann
 *
 * This file is part of JPMML-SkLearn
 *
 * JPMML-SkLearn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SkLearn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SkLearn.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sklearn;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.Batch;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.evaluator.testing.RealNumberEquivalence;
import org.junit.Test;
import sklearn.Estimator;

public class ClassifierTest extends SkLearnTest implements Algorithms, Datasets {

	@Override
	protected Batch createBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		Batch result = new SkLearnTestBatch(name, dataset, predicate, equivalence){

			@Override
			public ClassifierTest getIntegrationTest(){
				return ClassifierTest.this;
			}

			@Override
			public List<Map<String, String>> getInput() throws IOException {
				String dataset = super.getDataset();

				if(dataset.endsWith("Cat")){
					dataset = dataset.substring(0, dataset.length() - "Cat".length());
				} else

				if(dataset.endsWith("Dict")){
					dataset = dataset.substring(0, dataset.length() - "Dict".length());
				}

				return loadRecords("/csv/" + dataset + ".csv");
			}
		};

		return result;
	}

	@Test
	public void evaluateDurationInDaysApollo() throws Exception {
		evaluate("DurationInDays", APOLLO);
	}

	@Test
	public void evaluateDurationInSecondsApollo() throws Exception {
		evaluate("DurationInSeconds", APOLLO);
	}

	@Test
	public void evaluateDecisionTreeAudit() throws Exception {
		evaluate(DECISION_TREE, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeAuditDict() throws Exception {
		evaluate(DECISION_TREE, AUDIT_DICT);
	}

	@Test
	public void evaluateDecisionTreeAuditNA() throws Exception {
		String[] transformFields = {FieldNameUtil.create("eval", "nodeId")};

		evaluate(DECISION_TREE, AUDIT_NA, excludeFields(transformFields));
	}

	@Test
	public void evaluateBalancedDecisionTreeEnsembleAudit() throws Exception {
		evaluate(BALANCED_DECISION_TREE_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeEnsembleAudit() throws Exception {
		evaluate(DECISION_TREE_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateDummyAudit() throws Exception {
		evaluate(DUMMY, AUDIT);
	}

	@Test
	public void evaluatExtraTreesAudit() throws Exception {
		evaluate(EXTRA_TREES, AUDIT);
	}

	@Test
	public void evaluateGBDTLRAudit() throws Exception {
		evaluate(GBDT_LR, AUDIT);
	}

	@Test
	public void evaluateGradientBoostingAudit() throws Exception {
		evaluate(GRADIENT_BOOSTING, AUDIT);
	}

	@Test
	public void evaluateHistGradientBoostingAudit() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING, AUDIT);
	}

	@Test
	public void evaluateHistGradientBoostingAuditNA() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING, AUDIT_NA);
	}

	@Test
	public void evaluateLGBMAudit() throws Exception {
		evaluate(LGBM, AUDIT, new RealNumberEquivalence(2));
	}

	@Test
	public void evaluateLGBMAuditCat() throws Exception {
		evaluate(LGBM, AUDIT_CAT, new RealNumberEquivalence(2));
	}

	@Test
	public void evaluateLGBMLRAuditCat() throws Exception {
		evaluate(LGBM_LR, AUDIT_CAT);
	}

	@Test
	public void evaluateLinearDiscriminantAnalysisAudit() throws Exception {
		evaluate(LINEAR_DISCRIMINANT_ANALYSIS, AUDIT);
	}

	@Test
	public void evaluateLinearSVCAudit() throws Exception {
		evaluate(LINEAR_SVC, AUDIT);
	}

	@Test
	public void evaluateMultinomialLogisticRegressionAudit() throws Exception {
		evaluate(MULTINOMIAL_LOGISTIC_REGRESSION, AUDIT);
	}

	@Test
	public void evaluateOvRLogisticRegressionAudit() throws Exception {
		evaluate(OVR_LOGISTIC_REGRESSION, AUDIT);
	}

	@Test
	public void evaluateLogisticRegressionAuditDict() throws Exception {
		evaluate(LOGISTIC_REGRESSION, AUDIT_DICT);
	}

	@Test
	public void evaluateLogisticRegressionAuditNA() throws Exception {
		String[] transformFields = {FieldNameUtil.create("eval", AUDIT_PROBABILITY_TRUE)};

		evaluate(LOGISTIC_REGRESSION, AUDIT_NA, excludeFields(transformFields));
	}

	@Test
	public void evaluateLogisticRegressionEnsembleAudit() throws Exception {
		evaluate(LOGISTIC_REGRESSION_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateNaiveBayesAudit() throws Exception {
		evaluate(NAIVE_BAYES, AUDIT, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateOneVsRestAudit() throws Exception {
		evaluate(ONE_VS_REST, AUDIT);
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		evaluate(RANDOM_FOREST, AUDIT);
	}

	@Test
	public void evaluateBalancedRandomForestAudit() throws Exception {
		evaluate(BALANCED_RANDOM_FOREST, AUDIT);
	}

	@Test
	public void evaluateRidgeAudit() throws Exception {
		evaluate(RIDGE, AUDIT);
	}

	@Test
	public void evaluateRidgeEnsembleAudit() throws Exception {
		evaluate(RIDGE_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateStackingEnsembleAudit() throws Exception {
		evaluate(STACKING_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateSVCAudit() throws Exception {
		evaluate(SVC, AUDIT);
	}

	@Test
	public void evaluateVotingEnsembleAudit() throws Exception {
		evaluate(VOTING_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateXGBAudit() throws Exception {
		evaluate(XGB, AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(8));
	}

	@Test
	public void evaluateXGBAuditNA() throws Exception {
		String[] transformFields = {AUDIT_PROBABILITY_FALSE, FieldNameUtil.create(Estimator.FIELD_PREDICT, AUDIT_ADJUSTED), FieldNameUtil.create("eval", AUDIT_ADJUSTED)};

		evaluate(XGB, AUDIT_NA, excludeFields(transformFields), new FloatEquivalence(8));
	}

	@Test
	public void evaluateXGBLRAudit() throws Exception {
		evaluate(XGB_LR, AUDIT);
	}

	@Test
	public void evaluateXGBRFAudit() throws Exception {
		evaluate(XGBRF, AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(4));
	}

	@Test
	public void evaluateXGBRFLRAudit() throws Exception {
		evaluate(XGBRF_LR, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeIris() throws Exception {
		evaluate(DECISION_TREE, IRIS);
	}

	@Test
	public void evaluateDecisionTreeEnsembleIris() throws Exception {
		evaluate(DECISION_TREE_ENSEMBLE, IRIS);
	}

	@Test
	public void evaluateDummyIris() throws Exception {
		evaluate(DUMMY, IRIS);
	}

	@Test
	public void evaluateExtraTreesIris() throws Exception {
		evaluate(EXTRA_TREES, IRIS);
	}

	@Test
	public void evaluateGradientBoostingIris() throws Exception {
		evaluate(GRADIENT_BOOSTING, IRIS);
	}

	@Test
	public void evaluateHistGradientBoostingIris() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING, IRIS);
	}

	@Test
	public void evaluateKNNIris() throws Exception {
		evaluate(KNN, IRIS, excludeFields(createNeighborFields(5)));
	}

	@Test
	public void evaluateLGBMIris() throws Exception {
		evaluate(LGBM, IRIS, new RealNumberEquivalence(1));
	}

	@Test
	public void evaluateLinearDiscriminantAnalysisIris() throws Exception {
		evaluate(LINEAR_DISCRIMINANT_ANALYSIS, IRIS);
	}

	@Test
	public void evaluateLinearSVCIris() throws Exception {
		evaluate(LINEAR_SVC, IRIS);
	}

	@Test
	public void evaluateMultinomialLogisticRegressionIris() throws Exception {
		evaluate(MULTINOMIAL_LOGISTIC_REGRESSION, IRIS);
	}

	@Test
	public void evaluateOvRLogisticRegressionIris() throws Exception {
		evaluate(OVR_LOGISTIC_REGRESSION, IRIS);
	}

	@Test
	public void evaluateLogisticRegressionEnsembleIris() throws Exception {
		evaluate(LOGISTIC_REGRESSION_ENSEMBLE, IRIS);
	}

	@Test
	public void evaluateMLPIris() throws Exception {
		evaluate(MLP, IRIS);
	}

	@Test
	public void evaluateNaiveBayesIris() throws Exception {
		evaluate(NAIVE_BAYES, IRIS);
	}

	@Test
	public void evaluateOneVsRestIris() throws Exception {
		evaluate(ONE_VS_REST, IRIS);
	}

	@Test
	public void evaluateRandomForestIris() throws Exception {
		evaluate(RANDOM_FOREST, IRIS);
	}

	@Test
	public void evaluateRidgeIris() throws Exception {
		evaluate(RIDGE, IRIS);
	}

	@Test
	public void evaluateRidgeEnsembleIris() throws Exception {
		evaluate(RIDGE_ENSEMBLE, IRIS);
	}

	@Test
	public void evaluateRuleSetIris() throws Exception {
		evaluate(RULE_SET, IRIS);
	}

	@Test
	public void evaluateSelectFirstIris() throws Exception {
		evaluate(SELECT_FIRST, IRIS, excludeFields(IRIS_PROBABILITY_SETOSA, IRIS_PROBABILITY_VERSICOLOR, IRIS_PROBABILITY_VIRGINICA));
	}

	@Test
	public void evaluateSGDIris() throws Exception {
		evaluate(SGD, IRIS);
	}

	@Test
	public void evaluateSGDLogIris() throws Exception {
		evaluate(SGD_LOG, IRIS);
	}

	@Test
	public void evaluateStackingEnsembleIris() throws Exception {
		evaluate(STACKING_ENSEMBLE, IRIS);
	}

	@Test
	public void evaluateSVCIris() throws Exception {
		evaluate(SVC, IRIS);
	}

	@Test
	public void evaluateNuSVCIris() throws Exception {
		evaluate(NU_SVC, IRIS);
	}

	@Test
	public void evaluateVotingEnsembleIris() throws Exception {
		evaluate(VOTING_ENSEMBLE, IRIS, excludeFields(IRIS_PROBABILITY_SETOSA, IRIS_PROBABILITY_VERSICOLOR, IRIS_PROBABILITY_VIRGINICA));
	}

	@Test
	public void evaluateXGBIris() throws Exception {
		evaluate(XGB, IRIS, excludeFields(IRIS_PROBABILITY_SETOSA), new FloatEquivalence(16));
	}

	@Test
	public void evaluateLinearSVCSentiment() throws Exception {
		evaluate(LINEAR_SVC, SENTIMENT);
	}

	@Test
	public void evaluateLogisticRegressionSentiment() throws Exception {
		evaluate(LOGISTIC_REGRESSION, SENTIMENT);
	}

	@Test
	public void evaluateRandomForestSentiment() throws Exception {
		evaluate(RANDOM_FOREST, SENTIMENT);
	}

	@Test
	public void evaluateDecisionTreeVersicolor() throws Exception {
		evaluate(DECISION_TREE, VERSICOLOR);
	}

	@Test
	public void evaluateDummyVersicolor() throws Exception {
		evaluate(DUMMY, VERSICOLOR);
	}

	@Test
	public void evaluateGBDTLRVersicolor() throws Exception {
		evaluate(GBDT_LR, VERSICOLOR);
	}

	@Test
	public void evaluateKNNVersicolor() throws Exception {
		evaluate(KNN, VERSICOLOR, excludeFields(createNeighborFields(5)));
	}

	@Test
	public void evaluateMLPVersicolor() throws Exception {
		evaluate(MLP, VERSICOLOR);
	}

	@Test
	public void evaluateSGDVersicolor() throws Exception {
		evaluate(SGD, VERSICOLOR);
	}

	@Test
	public void evaluateSGDLogVersicolor() throws Exception {
		evaluate(SGD_LOG, VERSICOLOR);
	}

	@Test
	public void evaluateSVCVersicolor() throws Exception {
		evaluate(SVC, VERSICOLOR);
	}

	@Test
	public void evaluateNuSVCVersicolor() throws Exception {
		evaluate(NU_SVC, VERSICOLOR);
	}

	@Test
	public void evaluateXGBRFLRVersicolor() throws Exception {
		evaluate(XGBRF_LR, VERSICOLOR);
	}

	static
	private String[] createNeighborFields(int count){
		String[] result = new String[count];

		for(int i = 0; i < count; i++){
			result[i] = FieldNameUtil.create("neighbor", String.valueOf(i + 1));
		}

		return result;
	}
}