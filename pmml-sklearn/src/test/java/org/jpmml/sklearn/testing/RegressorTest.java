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
package org.jpmml.sklearn.testing;

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.ResultField;
import org.jpmml.model.visitors.VisitorBattery;
import org.jpmml.sklearn.FieldNames;
import org.junit.Test;
import sklearn.Estimator;

public class RegressorTest extends ValidatingSkLearnEncoderBatchTest implements SkLearnAlgorithms, Datasets, Fields {

	@Override
	public ValidatingSkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		ValidatingSkLearnEncoderBatch result = new ValidatingSkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public RegressorTest getArchiveBatchTest(){
				return RegressorTest.this;
			}

			@Override
			public VisitorBattery getValidators(){
				VisitorBattery visitorBattery = super.getValidators();

				String algorithm = getAlgorithm();
				String dataset = getDataset();

				if((AUTO).equals(dataset) || (AUTO_NA).equals(dataset) || (HOUSING).equals(dataset) || (VISIT).equals(dataset)){

					switch(algorithm){
						case DUMMY:
						case ISOTONIC_REGRESSION_INCR:
						case ISOTONIC_REGRESSION_DECR:
							break;
						default:
							visitorBattery.add(ModelStatsInspector.class);
							break;
					}
				}

				return visitorBattery;
			}
		};

		return result;
	}

	@Test
	public void evaluateAdaBoostAuto() throws Exception {
		evaluate(ADA_BOOST, AUTO);
	}

	@Test
	public void evaluateBayesianARDAuto() throws Exception {
		evaluate(BAYESIAN_ARD, AUTO);
	}

	@Test
	public void evaluateBayesianRidgeAuto() throws Exception {
		evaluate(BAYESIAN_RIDGE, AUTO);
	}

	@Test
	public void evaluateCQRAuto() throws Exception {
		evaluate(CQR, AUTO);
	}

	@Test
	public void evaluateDecisionTreeAuto() throws Exception {
		evaluate(DECISION_TREE, AUTO);
	}

	@Test
	public void evaluateDecisionTreeAutoNA() throws Exception {
		String[] transformFields = {FieldNameUtil.create("eval", FieldNames.NODE_ID)};

		evaluate(DECISION_TREE, AUTO_NA, excludeFields(transformFields));
	}

	@Test
	public void evaluateDecisionTreeEnsembleAuto() throws Exception {
		evaluate(DECISION_TREE_ENSEMBLE, AUTO);
	}

	@Test
	public void evaluateTransformedDecisionTreeAuto() throws Exception {
		evaluate(TRANSFORMED_DECISION_TREE, AUTO);
	}

	@Test
	public void evaluateDummyAuto() throws Exception {
		evaluate(DUMMY, AUTO);
	}

	@Test
	public void evaluateElasticNetAuto() throws Exception {
		evaluate(ELASTIC_NET, AUTO);
	}

	@Test
	public void evaluateExtraTreesAuto() throws Exception {
		evaluate(EXTRA_TREES, AUTO);
	}

	@Test
	public void evaluateMultiEstimatorChainAuto() throws Exception {
		evaluate(MULTI_ESTIMATOR_CHAIN, AUTO, excludeFields(FieldNameUtil.create(Estimator.FIELD_PREDICT, "acceleration"), FieldNames.NODE_ID));
	}

	@Test
	public void evaluateGBDTLMAuto() throws Exception {
		evaluate(GBDT_LM, AUTO);
	}

	@Test
	public void evaluateGradientBoostingAuto() throws Exception {
		evaluate(GRADIENT_BOOSTING, AUTO);
	}

	@Test
	public void evaluateHistGradientBoostingAuto() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING, AUTO);
	}

	@Test
	public void evaluateHistGradientBoostingAutoNA() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING, AUTO_NA);
	}

	@Test
	public void evaluateHuberAuto() throws Exception {
		evaluate(HUBER, AUTO);
	}

	@Test
	public void evaluateIsotonicRegressionIncrAuto() throws Exception {
		evaluate(ISOTONIC_REGRESSION_INCR, AUTO);
	}

	@Test
	public void evaluateIsotonicRegressionDecrAuto() throws Exception {
		evaluate(ISOTONIC_REGRESSION_DECR, AUTO);
	}

	@Test
	public void evaluateMultiKNNAuto() throws Exception {
		evaluate(MULTI_KNN, AUTO, excludeFields(createNeighborFields(5)));
	}

	@Test
	public void evaluateLarsAuto() throws Exception {
		evaluate(LARS, AUTO);
	}

	@Test
	public void evaluateLassoAuto() throws Exception {
		evaluate(LASSO, AUTO);
	}

	@Test
	public void evaluateLassoLarsAuto() throws Exception {
		evaluate(LASSO_LARS, AUTO);
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate(LINEAR_REGRESSION, AUTO);
	}

	@Test
	public void evaluateLinearRegressionAutoNA() throws Exception {
		String[] transformFields = {FieldNameUtil.create(Estimator.FIELD_PREDICT, AUTO_MPG), FieldNameUtil.create("cut", FieldNameUtil.create(Estimator.FIELD_PREDICT, AUTO_MPG))};

		evaluate(LINEAR_REGRESSION, AUTO_NA, excludeFields(transformFields));
	}

	@Test
	public void evaluateLinearRegressionChain() throws Exception {
		evaluate(LINEAR_REGRESSION_CHAIN, AUTO, excludeFields(FieldNameUtil.create(Estimator.FIELD_PREDICT, "acceleration")));
	}

	@Test
	public void evaluateLinearRegressionEnsembleAuto() throws Exception {
		evaluate(LINEAR_REGRESSION_ENSEMBLE, AUTO);
	}

	@Test
	public void evaluateMultiLinearRegressionAuto() throws Exception {
		evaluate(MULTI_LINEAR_REGRESSION, AUTO);
	}

	@Test
	public void evaluateTransformedLinearRegressionAuto() throws Exception {
		evaluate(TRANSFORMED_LINEAR_REGRESSION, AUTO);
	}

	@Test
	public void evaluateMultiMLPAuto() throws Exception {
		evaluate(MULTI_MLP, AUTO);
	}

	@Test
	public void evaluateOMPAuto() throws Exception {
		evaluate(OMP, AUTO);
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate(RANDOM_FOREST, AUTO);
	}

	@Test
	public void evaluateRidgeAuto() throws Exception {
		evaluate(RIDGE, AUTO);
	}

	@Test
	public void evaluateStackingEnsembleAuto() throws Exception {
		evaluate(STACKING_ENSEMBLE, AUTO);
	}

	@Test
	public void evaluateMultiLinearSVRAuto() throws Exception {
		evaluate(MULTI_LINEAR_SVR, AUTO);
	}

	@Test
	public void evaluateTheilSenAuto() throws Exception {
		evaluate(THEIL_SEN, AUTO);
	}

	@Test
	public void evaluateVotingEnsembleAuto() throws Exception {
		evaluate(VOTING_ENSEMBLE, AUTO);
	}

	@Test
	public void evaluateAdaBoostHousing() throws Exception {
		evaluate(ADA_BOOST, HOUSING);
	}

	@Test
	public void evaluateBayesianRidgeHousing() throws Exception {
		evaluate(BAYESIAN_RIDGE, HOUSING);
	}

	@Test
	public void evaluateGBDTLMHousing() throws Exception {
		evaluate(GBDT_LM, HOUSING);
	}

	@Test
	public void evaluateHistGradientBoostingHousing() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING, HOUSING);
	}

	@Test
	public void evaluateKNNHousing() throws Exception {
		evaluate(KNN, HOUSING);
	}

	@Test
	public void evaluateMLPHousing() throws Exception {
		evaluate(MLP, HOUSING);
	}

	@Test
	public void evaluateSGDHousing() throws Exception {
		evaluate(SGD, HOUSING);
	}

	@Test
	public void evaluateSVRHousing() throws Exception {
		evaluate(SVR, HOUSING);
	}

	@Test
	public void evaluateLinearSVRHousing() throws Exception {
		evaluate(LINEAR_SVR, HOUSING);
	}

	@Test
	public void evaluateNuSVRHousing() throws Exception {
		evaluate(NU_SVR, HOUSING);
	}

	@Test
	public void evaluateVotingEnsembleHousing() throws Exception {
		evaluate(VOTING_ENSEMBLE, HOUSING);
	}

	@Test
	public void evaluateGammaRegressionVisit() throws Exception {
		evaluate(GAMMA_REGRESSION, VISIT);
	}

	@Test
	public void evaluatePoissonRegressionVisit() throws Exception {
		evaluate(POISSON_REGRESSION, VISIT);
	}
}