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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.FieldNames;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.model.visitors.VisitorBattery;
import org.junit.jupiter.api.Test;
import sklearn.Estimator;
import sklearn.tree.HasTreeOptions;

public class ClassifierTest extends ValidatingSkLearnEncoderBatchTest implements SkLearnAlgorithms, Datasets, Fields {

	@Override
	public ValidatingSkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		ValidatingSkLearnEncoderBatch result = new ValidatingSkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public ClassifierTest getArchiveBatchTest(){
				return ClassifierTest.this;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				String algorithm = getAlgorithm();
				String dataset = getDataset();

				if(Objects.equals(AUDIT, dataset) || Objects.equals(IRIS, dataset)){

					if(Objects.equals(DECISION_TREE, algorithm) || Objects.equals(RANDOM_FOREST, algorithm)){
						Map<String, Object> options = new LinkedHashMap<>();
						options.put(HasTreeOptions.OPTION_INPUT_FLOAT, new Boolean[]{false, true});

						return OptionsUtil.generateOptionsMatrix(options);
					}
				} else

				if(Objects.equals(AUDIT_NA, dataset) || Objects.equals(IRIS_NA, dataset)){

					if(Objects.equals(RANDOM_FOREST, algorithm)){
						Map<String, Object> options = new LinkedHashMap<>();
						options.put(HasTreeOptions.OPTION_ALLOW_MISSING, Boolean.TRUE);
						options.put(HasTreeOptions.OPTION_COMPACT, new Boolean[]{false, true});

						return OptionsUtil.generateOptionsMatrix(options);
					}
				}

				return super.getOptionsMatrix();
			}

			@Override
			public String getInputCsvPath(){
				String path = super.getInputCsvPath();

				path = path.replace("Dict", "");

				return path;
			}

			@Override
			public VisitorBattery getValidators(){
				VisitorBattery visitorBattery = super.getValidators();

				String algorithm = getAlgorithm();
				String dataset = getDataset();

				modelStats:
				if(Objects.equals(AUDIT, dataset) || Objects.equals(AUDIT_NA, dataset) || Objects.equals(IRIS, dataset) || Objects.equals(IRIS_NA, dataset)){

					switch(algorithm){
						case DUMMY:
						case "Frozen":
							break;
						default:
							visitorBattery.add(ModelStatsInspector.class);
							break;
					}
				} // End if

				modelExplanation:
				if(Objects.equals(AUDIT, dataset)){

					if(algorithm.startsWith("Multi")){
						break modelExplanation;
					}

					switch(algorithm){
						case LOGISTIC_REGRESSION_CHAIN:
							break;
						default:
							visitorBattery.add(ModelExplanationInspector.class);
							break;
					}
				} // End if

				if(Objects.equals(AUDIT, dataset)){

					switch(algorithm){
						case DECISION_TREE:
						//case EXTRA_TREES:
						case GRADIENT_BOOSTING:
						//case RANDOM_FOREST:
							visitorBattery.add(FeatureImportancesInspector.class);
							break;
						default:
							break;
					}
				}

				return visitorBattery;
			}
		};

		return result;
	}

	@Test
	public void evaluateAdaBoostAudit() throws Exception {
		evaluate(ADA_BOOST, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeAudit() throws Exception {
		evaluate(DECISION_TREE, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeAuditDict() throws Exception {
		evaluate(DECISION_TREE, AUDIT + "Dict");
	}

	@Test
	public void evaluateDecisionTreeAuditNA() throws Exception {
		String[] transformFields = {FieldNameUtil.create("eval", FieldNames.NODE_ID)};

		evaluate(DECISION_TREE, AUDIT_NA, excludeFields(transformFields));
	}

	@Test
	public void evaluateDecisionTreeEnsembleAudit() throws Exception {
		evaluate(DECISION_TREE_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeIsotonicAudit() throws Exception {
		evaluate(DECISION_TREE + "Isotonic", AUDIT);
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
	public void evaluateGaussianNBAudit() throws Exception {
		evaluate(GAUSSIAN_NB, AUDIT, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateGradientBoostingSigmoidAudit() throws Exception {
		evaluate(GRADIENT_BOOSTING + "Sigmoid", AUDIT);
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
	public void evaluateMultiKNNAudit() throws Exception {
		evaluate(MULTI_KNN, AUDIT);
	}

	@Test
	public void evaluateLinearDiscriminantAnalysisAudit() throws Exception {
		evaluate(LINEAR_DISCRIMINANT_ANALYSIS, AUDIT, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateLinearSVCAudit() throws Exception {
		evaluate(LINEAR_SVC, AUDIT);
	}

	@Test
	public void evaluateLinearSVCIsotonicAudit() throws Exception {
		evaluate(LINEAR_SVC + "Isotonic", AUDIT, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateMultiLogisticRegression() throws Exception {
		String[] resultFields = {
			FieldNameUtil.create(FieldNames.PROBABILITY, "Gender", "Male"), FieldNameUtil.create(FieldNames.PROBABILITY, "Gender", "Female"),
			FieldNameUtil.create(FieldNames.PROBABILITY, "Adjusted", 0), FieldNameUtil.create(FieldNames.PROBABILITY, "Adjusted", 1)
		};

		evaluate(MULTI_LOGISTIC_REGRESSION, AUDIT, excludeFields(resultFields));
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		evaluate(LOGISTIC_REGRESSION, AUDIT);
	}

	@Test
	public void evaluateLogisticRegressionAuditDict() throws Exception {
		evaluate(LOGISTIC_REGRESSION, AUDIT + "Dict");
	}

	@Test
	public void evaluateLogisticRegressionAuditNA() throws Exception {
		String[] transformFields = {FieldNameUtil.create("eval", AUDIT_PROBABILITY_TRUE)};

		evaluate(LOGISTIC_REGRESSION, AUDIT_NA, excludeFields(transformFields));
	}

	@Test
	public void evaluateLogisticRegressionChainAudit() throws Exception {
		String[] resultFields = {
			FieldNameUtil.create(Estimator.FIELD_PREDICT, "Gender"), FieldNameUtil.create(FieldNames.PROBABILITY, "Gender", -2), FieldNameUtil.create(FieldNames.PROBABILITY, "Gender", 2),
			FieldNameUtil.create(FieldNames.PROBABILITY, "Adjusted", 0), FieldNameUtil.create(FieldNames.PROBABILITY, "Adjusted", 1)
		};

		evaluate(LOGISTIC_REGRESSION_CHAIN, AUDIT, excludeFields(resultFields));
	}

	@Test
	public void evaluateLogisticRegressionEnsembleAudit() throws Exception {
		evaluate(LOGISTIC_REGRESSION_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateNearestCentroidAuditDict() throws Exception {
		evaluate(NEAREST_CENTROID, AUDIT + "Dict");
	}

	@Test
	public void evaluateOneHotEncoderAuditNA() throws Exception {
		evaluate("OneHotEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateOneVsRestAudit() throws Exception {
		evaluate(ONE_VS_REST, AUDIT);
	}

	@Test
	public void evaluateOrdinalEncoderAuditNA() throws Exception {
		evaluate("OrdinalEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		evaluate(RANDOM_FOREST, AUDIT);
	}

	@Test
	public void evaluateRandomForestSigmoidAudit() throws Exception {
		evaluate(RANDOM_FOREST + "Sigmoid", AUDIT);
	}

	@Test
	public void evaluateRandomForestAuditNA() throws Exception {
		evaluate(RANDOM_FOREST, AUDIT_NA);
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
	public void evaluateRidgeSigmoidAudit() throws Exception {
		evaluate(RIDGE + "Sigmoid", AUDIT);
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
	public void evaluateTargetEncoderAuditNA() throws Exception {
		evaluate("TargetEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateVotingEnsembleAudit() throws Exception {
		evaluate(VOTING_ENSEMBLE, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeIris() throws Exception {
		evaluate(DECISION_TREE, IRIS);
	}

	@Test
	public void evaluateDecisionTreeIrisNA() throws Exception {
		evaluate(DECISION_TREE, IRIS_NA);
	}

	@Test
	public void evaluateDecisionTreeEnsembleIris() throws Exception {
		evaluate(DECISION_TREE_ENSEMBLE, IRIS);
	}

	@Test
	public void evaluateDecisionTreeIsotonicIris() throws Exception {
		evaluate(DECISION_TREE + "Isotonic", IRIS);
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
	public void evaluateFrozenIris() throws Exception {
		evaluate("Frozen", IRIS);
	}

	@Test
	public void evaluateGaussianNBIris() throws Exception {
		evaluate(GAUSSIAN_NB, IRIS);
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
	public void evaluateLinearDiscriminantAnalysisIris() throws Exception {
		evaluate(LINEAR_DISCRIMINANT_ANALYSIS, IRIS);
	}

	@Test
	public void evaluateLinearSVCIris() throws Exception {
		evaluate(LINEAR_SVC, IRIS);
	}

	@Test
	public void evaluateLinearSVCIsotonicIris() throws Exception {
		evaluate(LINEAR_SVC + "Isotonic", IRIS, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate(LOGISTIC_REGRESSION, IRIS);
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
	public void evaluateOneVsRestIris() throws Exception {
		evaluate(ONE_VS_REST, IRIS);
	}

	@Test
	public void evaluatePerceptronIris() throws Exception {
		evaluate(PERCEPTRON, IRIS);
	}

	@Test
	public void evaluateRandomForestIris() throws Exception {
		evaluate(RANDOM_FOREST, IRIS);
	}

	@Test
	public void evaluateRandomForestIrisNA() throws Exception {
		evaluate(RANDOM_FOREST, IRIS_NA);
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
	public void evaluateRidgeSigmoidIris() throws Exception {
		evaluate(RIDGE + "Sigmoid", IRIS);
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
	public void evaluateSGDSigmoidIris() throws Exception {
		evaluate(SGD + "Sigmoid", IRIS);
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
	public void evaluateBernoulliNBSentiment() throws Exception {
		evaluate(BERNOULLI_NB, SENTIMENT);
	}

	@Test
	public void evaluateBernoulliNBSmoothSentiment() throws Exception {
		evaluate(BERNOULLI_NB + "Smooth", SENTIMENT);
	}

	@Test
	public void evaluateCategoricalNBSentiment() throws Exception {
		evaluate(CATEGORICAL_NB, SENTIMENT);
	}

	@Test
	public void evaluateCategoricalNBSmoothSentiment() throws Exception {
		evaluate(CATEGORICAL_NB + "Smooth", SENTIMENT);
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
	public void evaluateMultinomialNBSentiment() throws Exception {
		evaluate(MULTINOMIAL_NB, SENTIMENT);
	}

	@Test
	public void evaluateMultinomialNBSMoothSentiment() throws Exception {
		evaluate(MULTINOMIAL_NB + "Smooth", SENTIMENT);
	}

	@Test
	public void evaluateRandomForestSentiment() throws Exception {
		evaluate(RANDOM_FOREST, SENTIMENT);
	}

	@Test
	public void evaluateAdaBoostVersicolor() throws Exception {
		evaluate(ADA_BOOST, VERSICOLOR);
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
	public void evaluateKNNVersicolor() throws Exception {
		evaluate(KNN, VERSICOLOR, excludeFields(createNeighborFields(5)));
	}

	@Test
	public void evaluateLogisticRegressionVersicolor() throws Exception {
		evaluate(LOGISTIC_REGRESSION, VERSICOLOR, excludeFields(IRIS_SPECIES));
	}

	@Test
	public void evaluateMLPVersicolor() throws Exception {
		evaluate(MLP, VERSICOLOR);
	}

	@Test
	public void evaluatePerceptronVersicolor() throws Exception {
		evaluate(PERCEPTRON, VERSICOLOR);
	}

	@Test
	public void evaluateSGDVersicolor() throws Exception {
		evaluate(SGD, VERSICOLOR);
	}

	@Test
	public void evaluateSGDLogVersicolor() throws Exception {
		evaluate(SGD_LOG, VERSICOLOR, excludeFields(IRIS_SPECIES));
	}

	@Test
	public void evaluateSVCVersicolor() throws Exception {
		evaluate(SVC, VERSICOLOR);
	}

	@Test
	public void evaluateNuSVCVersicolor() throws Exception {
		evaluate(NU_SVC, VERSICOLOR);
	}
}