/*
 * Copyright (c) 2026 Villu Ruusmann
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
package org.jpmml.sklearn.extension.testing;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnAlgorithms;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;
import sklearn.tree.HasTreeOptions;

public class CausalMLTest extends SkLearnEncoderBatchTest implements SkLearnAlgorithms {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public CausalMLTest getArchiveBatchTest(){
				return CausalMLTest.this;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				String algorithm = getAlgorithm();

				if(algorithm.endsWith("SRegressor")){
					Map<String, Object> options = new LinkedHashMap<>();
					options.put(HasTreeOptions.OPTION_COMPACT, new Boolean[]{false, true});

					return OptionsUtil.generateOptionsMatrix(options);
				}

				return super.getOptionsMatrix();
			}

			@Override
			public Table getInput() throws IOException {
				Table table = super.getInput();

				String dataset = getDataset();

				if(Objects.equals(dataset, "EmailBin")){
					Function<Object, String> function = new Function<Object, String>(){

						@Override
						public String apply(Object object){
							String string = (String)object;

							if(Objects.equals("control", string)){
								return "control";
							} else

							{
								return "email";
							}
						}
					};

					table.apply("segment", function);
				}

				return table;
			}

			@Override
			public String getInputCsvPath(){
				String path = super.getInputCsvPath();

				path = path.replace("Bin", "");

				return path;
			}
		};

		return result;
	}

	@Test
	public void evaluateElasticNetPropensityModelEmail() throws Exception {
		evaluate(ELASTIC_NET + "PropensityModel", "Email");
	}

	@Test
	public void evaluateElasticNetPropensityModelIsotonicEmail() throws Exception {
		evaluate(ELASTIC_NET + "PropensityModel" + "Isotonic", "Email");
	}

	@Test
	public void evaluateLogisticRegressionPropensityModelEmail() throws Exception {
		evaluate(LOGISTIC_REGRESSION + "PropensityModel", "Email");
	}

	@Test
	public void evaluateLogisticRegressionPropensityModelIsotonicEmail() throws Exception {
		evaluate(LOGISTIC_REGRESSION + "PropensityModel" + "Isotonic", "Email");
	}

	@Test
	public void evaluateDecisionTreeSClassifierEmail() throws Exception {
		evaluate(DECISION_TREE + "SClassifier", "Email");
	}

	@Test
	public void evaluateDecisionTreeSClassifierEmailBin() throws Exception {
		evaluate(DECISION_TREE + "SClassifier", "EmailBin");
	}

	@Test
	public void evaluateDecisionTreeSRegressorEmail() throws Exception {
		evaluate(DECISION_TREE + "SRegressor", "Email");
	}

	@Test
	public void evaluateDecisionTreeSRegressorEmailBin() throws Exception {
		evaluate(DECISION_TREE + "SRegressor", "EmailBin");
	}

	@Test
	public void evaluateDecisionTreeTClassifierEmail() throws Exception {
		evaluate(DECISION_TREE + "TClassifier", "Email");
	}

	@Test
	public void evaluateDecisionTreeTClassifierEmailBin() throws Exception {
		evaluate(DECISION_TREE + "TClassifier", "EmailBin");
	}

	@Test
	public void evaluateDecisionTreeTRegressorEmail() throws Exception {
		evaluate(DECISION_TREE + "TRegressor", "Email");
	}

	@Test
	public void evaluateDecisionTreeTRegressorEmailBin() throws Exception {
		evaluate(DECISION_TREE + "TRegressor", "EmailBin");
	}

	@Test
	public void evaluateDecisionTreeXClassifierEmail() throws Exception {
		evaluate(DECISION_TREE + "XClassifier", "Email");
	}

	@Test
	public void evaluateDecisionTreeXClassifierEmailBin() throws Exception {
		evaluate(DECISION_TREE + "XClassifier", "EmailBin");
	}

	@Test
	public void evaluateDecisionTreeXRegressorEmail() throws Exception {
		evaluate(DECISION_TREE + "XRegressor", "Email");
	}

	@Test
	public void evaluateDecisionTreeXRegressorEmailBin() throws Exception {
		evaluate(DECISION_TREE + "XRegressor", "EmailBin");
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmail() throws Exception {
		evaluate(GRADIENT_BOOSTING + "SRegressor", "Email");
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmailBin() throws Exception {
		evaluate(GRADIENT_BOOSTING + "SRegressor", "EmailBin");
	}

	@Test
	public void evaluateGradientBoostingTRegressorEmail() throws Exception {
		evaluate(GRADIENT_BOOSTING + "TRegressor", "Email", new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateGradientBoostingTRegressorEmailBin() throws Exception {
		evaluate(GRADIENT_BOOSTING + "TRegressor", "EmailBin", new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateGradientBoostingXRegressorEmail() throws Exception {
		evaluate(GRADIENT_BOOSTING + "XRegressor", "Email", new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateGradientBoostingXRegressorEmailBin() throws Exception {
		evaluate(GRADIENT_BOOSTING + "XRegressor", "EmailBin");
	}

	@Test
	public void evaluateRandomForestSClassifierEmail() throws Exception {
		evaluate(RANDOM_FOREST + "SClassifier", "Email", new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateRandomForestSClassifierEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "SClassifier", "EmailBin", new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateRandomForestSRegressorEmail() throws Exception {
		evaluate(RANDOM_FOREST + "SRegressor", "Email", new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateRandomForestSRegressorEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "SRegressor", "EmailBin", new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateRandomForestTClassifierEmail() throws Exception {
		evaluate(RANDOM_FOREST + "TClassifier", "Email");
	}

	@Test
	public void evaluateRandomForestTClassifierEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "TClassifier", "EmailBin");
	}

	@Test
	public void evaluateRandomForestTRegressorEmail() throws Exception {
		evaluate(RANDOM_FOREST + "TRegressor", "Email");
	}

	@Test
	public void evaluateRandomForestTRegressorEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "TRegressor", "EmailBin");
	}

	@Test
	public void evaluateRandomForestXClassifierEmail() throws Exception {
		evaluate(RANDOM_FOREST + "XClassifier", "Email");
	}

	@Test
	public void evaluateRandomForestXClassifierEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "XClassifier", "EmailBin");
	}

	@Test
	public void evaluateRandomForestXRegressorEmail() throws Exception {
		evaluate(RANDOM_FOREST + "XRegressor", "Email");
	}

	@Test
	public void evaluateRandomForestXRegressorEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "XRegressor", "EmailBin");
	}
}