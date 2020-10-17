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
import org.dmg.pmml.FieldName;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.Batch;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.evaluator.testing.RealNumberEquivalence;
import org.junit.Test;

public class ClassifierTest extends SkLearnTest {

	@Override
	protected Batch createBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		Batch result = new SkLearnTestBatch(name, dataset, predicate, equivalence){

			@Override
			public ClassifierTest getIntegrationTest(){
				return ClassifierTest.this;
			}

			@Override
			public List<Map<FieldName, String>> getInput() throws IOException {
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
		evaluate("DurationInDays", "Apollo");
	}

	@Test
	public void evaluateDurationInSecondsApollo() throws Exception {
		evaluate("DurationInSeconds", "Apollo");
	}

	@Test
	public void evaluateDecisionTreeAudit() throws Exception {
		evaluate("DecisionTree", "Audit");
	}

	@Test
	public void evaluateDecisionTreeAuditDict() throws Exception {
		evaluate("DecisionTree", "AuditDict");
	}

	@Test
	public void evaluateDecisionTreeAuditNA() throws Exception {
		FieldName[] transformFields = {FieldName.create("eval(nodeId)")};

		evaluate("DecisionTree", "AuditNA", excludeFields(transformFields));
	}

	@Test
	public void evaluateBalancedDecisionTreeEnsembleAudit() throws Exception {
		evaluate("BalancedDecisionTreeEnsemble", "Audit");
	}

	@Test
	public void evaluateDecisionTreeEnsembleAudit() throws Exception {
		evaluate("DecisionTreeEnsemble", "Audit");
	}

	@Test
	public void evaluateDummyAudit() throws Exception {
		evaluate("Dummy", "Audit");
	}

	@Test
	public void evaluatExtraTreesAudit() throws Exception {
		evaluate("ExtraTrees", "Audit");
	}

	@Test
	public void evaluateGBDTLRAudit() throws Exception {
		evaluate("GBDTLR", "Audit");
	}

	@Test
	public void evaluateGradientBoostingAudit() throws Exception {
		evaluate("GradientBoosting", "Audit");
	}

	@Test
	public void evaluateHistGradientBoostingAudit() throws Exception {
		evaluate("HistGradientBoosting", "Audit");
	}

	@Test
	public void evaluateHistGradientBoostingAuditNA() throws Exception {
		evaluate("HistGradientBoosting", "AuditNA");
	}

	@Test
	public void evaluateLGBMAudit() throws Exception {
		evaluate("LGBM", "Audit", new RealNumberEquivalence(1));
	}

	@Test
	public void evaluateLGBMAuditCat() throws Exception {
		evaluate("LGBM", "AuditCat", new RealNumberEquivalence(2));
	}

	@Test
	public void evaluateLGBMLRAuditCat() throws Exception {
		evaluate("LGBMLR", "AuditCat");
	}

	@Test
	public void evaluateLinearDiscriminantAnalysisAudit() throws Exception {
		evaluate("LinearDiscriminantAnalysis", "Audit");
	}

	@Test
	public void evaluateLinearSVCAudit() throws Exception {
		evaluate("LinearSVC", "Audit");
	}

	@Test
	public void evaluateMultinomialLogisticRegressionAudit() throws Exception {
		evaluate("MultinomialLogisticRegression", "Audit");
	}

	@Test
	public void evaluateOvRLogisticRegressionAudit() throws Exception {
		evaluate("OvRLogisticRegression", "Audit");
	}

	@Test
	public void evaluateLogisticRegressionAuditDict() throws Exception {
		evaluate("LogisticRegression", "AuditDict");
	}

	@Test
	public void evaluateLogisticRegressionAuditNA() throws Exception {
		FieldName[] transformFields = {FieldName.create("eval(probability(1))")};

		evaluate("LogisticRegression", "AuditNA", excludeFields(transformFields));
	}

	@Test
	public void evaluateLogisticRegressionEnsembleAudit() throws Exception {
		evaluate("LogisticRegressionEnsemble", "Audit");
	}

	@Test
	public void evaluateNaiveBayesAudit() throws Exception {
		evaluate("NaiveBayes", "Audit", new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateOneVsRestAudit() throws Exception {
		evaluate("OneVsRest", "Audit");
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		evaluate("RandomForest", "Audit");
	}

	@Test
	public void evaluateBalancedRandomForestAudit() throws Exception {
		evaluate("BalancedRandomForest", "Audit");
	}

	@Test
	public void evaluateRidgeAudit() throws Exception {
		evaluate("Ridge", "Audit");
	}

	@Test
	public void evaluateRidgeEnsembleAudit() throws Exception {
		evaluate("RidgeEnsemble", "Audit");
	}

	@Test
	public void evaluateStackingEnsembleAudit() throws Exception {
		evaluate("StackingEnsemble", "Audit");
	}

	@Test
	public void evaluateSVCAudit() throws Exception {
		evaluate("SVC", "Audit");
	}

	@Test
	public void evaluateTPOTAudit() throws Exception {
		evaluate("TPOT", "Audit");
	}

	@Test
	public void evaluateVotingEnsembleAudit() throws Exception {
		evaluate("VotingEnsemble", "Audit");
	}

	@Test
	public void evaluateXGBAudit() throws Exception {
		evaluate("XGB", "Audit", excludeFields(ClassifierTest.falseProbabilityField), new FloatEquivalence(8));
	}

	@Test
	public void evaluateXGBAuditNA() throws Exception {
		FieldName[] transformFields = {ClassifierTest.falseProbabilityField, FieldName.create("predict(Adjusted)"), FieldName.create("eval(Adjusted)")};

		evaluate("XGB", "AuditNA", excludeFields(transformFields), new FloatEquivalence(8));
	}

	@Test
	public void evaluateXGBLRAudit() throws Exception {
		evaluate("XGBLR", "Audit");
	}

	@Test
	public void evaluateXGBRFAudit() throws Exception {
		evaluate("XGBRF", "Audit", excludeFields(ClassifierTest.falseProbabilityField), new FloatEquivalence(4));
	}

	@Test
	public void evaluateXGBRFLRAudit() throws Exception {
		evaluate("XGBRFLR", "Audit");
	}

	@Test
	public void evaluateDecisionTreeIris() throws Exception {
		evaluate("DecisionTree", "Iris");
	}

	@Test
	public void evaluateDecisionTreeEnsembleIris() throws Exception {
		evaluate("DecisionTreeEnsemble", "Iris");
	}

	@Test
	public void evaluateDummyIris() throws Exception {
		evaluate("Dummy", "Iris");
	}

	@Test
	public void evaluateExtraTreesIris() throws Exception {
		evaluate("ExtraTrees", "Iris");
	}

	@Test
	public void evaluateGradientBoostingIris() throws Exception {
		evaluate("GradientBoosting", "Iris");
	}

	@Test
	public void evaluateHistGradientBoostingIris() throws Exception {
		evaluate("HistGradientBoosting", "Iris");
	}

	@Test
	public void evaluateKNNIris() throws Exception {
		evaluate("KNN", "Iris", excludeFields(ClassifierTest.neighborFields));
	}

	@Test
	public void evaluateLGBMIris() throws Exception {
		evaluate("LGBM", "Iris", new RealNumberEquivalence(1));
	}

	@Test
	public void evaluateLinearDiscriminantAnalysisIris() throws Exception {
		evaluate("LinearDiscriminantAnalysis", "Iris");
	}

	@Test
	public void evaluateLinearSVCIris() throws Exception {
		evaluate("LinearSVC", "Iris");
	}

	@Test
	public void evaluateMultinomialLogisticRegressionIris() throws Exception {
		evaluate("MultinomialLogisticRegression", "Iris");
	}

	@Test
	public void evaluateOvRLogisticRegressionIris() throws Exception {
		evaluate("OvRLogisticRegression", "Iris");
	}

	@Test
	public void evaluateLogisticRegressionEnsembleIris() throws Exception {
		evaluate("LogisticRegressionEnsemble", "Iris");
	}

	@Test
	public void evaluateMLPIris() throws Exception {
		evaluate("MLP", "Iris");
	}

	@Test
	public void evaluateNaiveBayesIris() throws Exception {
		evaluate("NaiveBayes", "Iris");
	}

	@Test
	public void evaluateOneVsRestIris() throws Exception {
		evaluate("OneVsRest", "Iris");
	}

	@Test
	public void evaluateRandomForestIris() throws Exception {
		evaluate("RandomForest", "Iris");
	}

	@Test
	public void evaluateRidgeIris() throws Exception {
		evaluate("Ridge", "Iris");
	}

	@Test
	public void evaluateRidgeEnsembleIris() throws Exception {
		evaluate("RidgeEnsemble", "Iris");
	}

	@Test
	public void evaluateRuleSetIris() throws Exception {
		evaluate("RuleSet", "Iris");
	}

	@Test
	public void evaluateSelectFirstIris() throws Exception {
		evaluate("SelectFirst", "Iris", excludeFields(ClassifierTest.irisProbabilityFields));
	}

	@Test
	public void evaluateSGDIris() throws Exception {
		evaluate("SGD", "Iris");
	}

	@Test
	public void evaluateSGDLogIris() throws Exception {
		evaluate("SGDLog", "Iris");
	}

	@Test
	public void evaluateStackingEnsembleIris() throws Exception {
		evaluate("StackingEnsemble", "Iris");
	}

	@Test
	public void evaluateSVCIris() throws Exception {
		evaluate("SVC", "Iris");
	}

	@Test
	public void evaluateNuSVCIris() throws Exception {
		evaluate("NuSVC", "Iris");
	}

	@Test
	public void evaluateTPOTIris() throws Exception {
		evaluate("TPOT", "Iris");
	}

	@Test
	public void evaluateVotingEnsembleIris() throws Exception {
		evaluate("VotingEnsemble", "Iris", excludeFields(ClassifierTest.irisProbabilityFields));
	}

	@Test
	public void evaluateXGBIris() throws Exception {
		evaluate("XGB", "Iris", new FloatEquivalence(8));
	}

	@Test
	public void evaluateLinearSVCSentiment() throws Exception {
		evaluate("LinearSVC", "Sentiment");
	}

	@Test
	public void evaluateLogisticRegressionSentiment() throws Exception {
		evaluate("LogisticRegression", "Sentiment");
	}

	@Test
	public void evaluateRandomForestSentiment() throws Exception {
		evaluate("RandomForest", "Sentiment");
	}

	@Test
	public void evaluateDecisionTreeVersicolor() throws Exception {
		evaluate("DecisionTree", "Versicolor");
	}

	@Test
	public void evaluateDummyVersicolor() throws Exception {
		evaluate("Dummy", "Versicolor");
	}

	@Test
	public void evaluateGBDTLRVersicolor() throws Exception {
		evaluate("GBDTLR", "Versicolor");
	}

	@Test
	public void evaluateKNNVersicolor() throws Exception {
		evaluate("KNN", "Versicolor", excludeFields(ClassifierTest.neighborFields));
	}

	@Test
	public void evaluateMLPVersicolor() throws Exception {
		evaluate("MLP", "Versicolor");
	}

	@Test
	public void evaluateSGDVersicolor() throws Exception {
		evaluate("SGD", "Versicolor");
	}

	@Test
	public void evaluateSGDLogVersicolor() throws Exception {
		evaluate("SGDLog", "Versicolor");
	}

	@Test
	public void evaluateSVCVersicolor() throws Exception {
		evaluate("SVC", "Versicolor");
	}

	@Test
	public void evaluateNuSVCVersicolor() throws Exception {
		evaluate("NuSVC", "Versicolor");
	}

	@Test
	public void evaluateTPOTVersicolor() throws Exception {
		evaluate("TPOT", "Versicolor", new PMMLEquivalence(5e-13, 5e-13));
	}

	@Test
	public void evaluateXGBRFLRVersicolor() throws Exception {
		evaluate("XGBRFLR", "Versicolor");
	}

	static
	private FieldName[] createFields(String prefix, int count){
		FieldName[] result = new FieldName[count];

		for(int i = 0; i < count; i++){
			result[i] = FieldNameUtil.create(prefix, String.valueOf(i + 1));
		}

		return result;
	}

	private static final FieldName[] neighborFields = createFields("neighbor", 5);

	private static final FieldName falseProbabilityField = FieldNameUtil.create("probability", "0");
	private static final FieldName trueProbabilityField = FieldNameUtil.create("probability", "1");

	private static final FieldName[] irisProbabilityFields = {FieldNameUtil.create("probability", "setosa"), FieldNameUtil.create("probability", "versicolor"), FieldNameUtil.create("probability", "virginica")};
}