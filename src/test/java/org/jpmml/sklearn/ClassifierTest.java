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

import java.util.LinkedHashSet;
import java.util.Set;

import com.google.common.collect.ImmutableSet;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.Batch;
import org.junit.Test;

public class ClassifierTest extends EstimatorTest {

	@Test
	public void evaluateDecisionTreeAudit() throws Exception {
		evaluate("DecisionTree", "Audit");
	}

	@Test
	public void evaluateDecisionTreeEnsembleAudit() throws Exception {
		evaluate("DecisionTreeEnsemble", "Audit");
	}

	@Test
	public void evaluatExtraTreesAudit() throws Exception {
		evaluate("ExtraTrees", "Audit");
	}

	@Test
	public void evaluateGradientBoostingAudit() throws Exception {
		evaluate("GradientBoosting", "Audit");
	}

	@Test
	public void evaluateLGBMAudit() throws Exception {
		evaluate("LGBM", "Audit");
	}

	@Test
	public void evaluateLinearDiscriminantAnalysisAudit() throws Exception {
		evaluate("LinearDiscriminantAnalysis", "Audit");
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		evaluate("LogisticRegression", "Audit");
	}

	@Test
	public void evaluateLogisticRegressionEnsembleAudit() throws Exception {
		evaluate("LogisticRegressionEnsemble", "Audit");
	}

	@Test
	public void evaluateNaiveBayesAudit() throws Exception {
		evaluate("NaiveBayes", "Audit");
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		evaluate("RandomForest", "Audit");
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
	public void evaluateSVCAudit() throws Exception {
		evaluate("SVC", "Audit");
	}

	@Test
	public void evaluateVotingEnsembleAudit() throws Exception {
		evaluate("VotingEnsemble", "Audit");
	}

	@Test
	public void evaluateXGBAudit() throws Exception {

		try(Batch batch = createBatch("XGB", "Audit")){
			evaluate(batch, null, 1e-5, 1e-5);
		}
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
	public void evaluateExtraTreesIris() throws Exception {
		evaluate("ExtraTrees", "Iris");
	}

	@Test
	public void evaluateGradientBoostingIris() throws Exception {
		evaluate("GradientBoosting", "Iris");
	}

	@Test
	public void evaluateKNNIris() throws Exception {

		try(Batch batch = createBatch("KNN", "Iris")){
			Set<FieldName> ignoredFields = createFieldList("neighbor", 5);

			evaluate(batch, ignoredFields);
		}
	}

	@Test
	public void evaluateLinearDiscriminantAnalysisIris() throws Exception {
		evaluate("LinearDiscriminantAnalysis", "Iris");
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate("LogisticRegression", "Iris");
	}

	@Test
	public void evaluateLogisticRegressionEnsembleIris() throws Exception {
		evaluate("LogisticRegressionEnsemble", "Iris");
	}

	@Test
	public void evaluateNaiveBayesIris() throws Exception {
		evaluate("NaiveBayes", "Iris");
	}

	@Test
	public void evaluateMLPIris() throws Exception {
		evaluate("MLP", "Iris");
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
	public void evaluateSGDIris() throws Exception {
		evaluate("SGD", "Iris");
	}

	@Test
	public void evaluateSGDLogIris() throws Exception {
		evaluate("SGDLog", "Iris");
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
	public void evaluateVotingEnsembleIris() throws Exception {

		try(Batch batch = createBatch("VotingEnsemble", "Iris")){
			Set<FieldName> ignoredFields = ImmutableSet.of(FieldName.create("probability_setosa"), FieldName.create("probability_versicolor"), FieldName.create("probability_virginica"));

			evaluate(batch, ignoredFields);
		}
	}

	@Test
	public void evaluateXGBIris() throws Exception {

		try(Batch batch = createBatch("XGB", "Iris")){
			evaluate(batch, null, 1e-5, 1e-5);
		}
	}

	@Test
	public void evaluateDecisionTreeVersicolor() throws Exception {
		evaluate("DecisionTree", "Versicolor");
	}

	@Test
	public void evaluateKNNVersicolor() throws Exception {

		try(Batch batch = createBatch("KNN", "Versicolor")){
			Set<FieldName> ignoredFields = createFieldList("neighbor", 5);

			evaluate(batch, ignoredFields);
		}
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

	static
	private Set<FieldName> createFieldList(String prefix, int count){
		Set<FieldName> result = new LinkedHashSet<>();

		for(int i = 0; i < count; i++){
			result.add(FieldName.create(prefix + "_" + (i + 1)));
		}

		return result;
	}
}