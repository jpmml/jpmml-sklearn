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
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnAlgorithms;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;
import sklearn.tree.HasTreeOptions;

public class CausalMLTest extends SkLearnEncoderBatchTest implements Datasets, SkLearnAlgorithms {

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

				if((algorithm.endsWith("RClassifier") || algorithm.endsWith("RRegressor")) || (algorithm.endsWith("SClassifier") || algorithm.endsWith("SRegressor")) || (algorithm.endsWith("TClassifier") || algorithm.endsWith("TRegressor"))){
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

				if(Objects.equals(dataset, EMAIL + "Bin")){
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
		evaluate(ELASTIC_NET + "PropensityModel", EMAIL);
	}

	@Test
	public void evaluateElasticNetPropensityModelIsotonicEmail() throws Exception {
		evaluate(ELASTIC_NET + "PropensityModel" + "Isotonic", EMAIL);
	}

	@Test
	public void evaluateLogisticRegressionPropensityModelEmail() throws Exception {
		evaluate(LOGISTIC_REGRESSION + "PropensityModel", EMAIL);
	}

	@Test
	public void evaluateLogisticRegressionPropensityModelIsotonicEmail() throws Exception {
		evaluate(LOGISTIC_REGRESSION + "PropensityModel" + "Isotonic", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeRClassifierEmail() throws Exception {
		evaluate(DECISION_TREE + "RClassifier", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeRClassifierEmailBin() throws Exception {
		evaluate(DECISION_TREE + "RClassifier", EMAIL + "Bin");
	}

	@Test
	public void evaluateDecisionTreeRRegressorEmail() throws Exception {
		evaluate(DECISION_TREE + "RRegressor", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeRRegressorEmailBin() throws Exception {
		evaluate(DECISION_TREE + "RRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateDecisionTreeSClassifierEmail() throws Exception {
		evaluate(DECISION_TREE + "SClassifier", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeSClassifierEmailBin() throws Exception {
		evaluate(DECISION_TREE + "SClassifier", EMAIL + "Bin");
	}

	@Test
	public void evaluateDecisionTreeSRegressorEmail() throws Exception {
		evaluate(DECISION_TREE + "SRegressor", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeSRegressorEmailBin() throws Exception {
		evaluate(DECISION_TREE + "SRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateDecisionTreeTClassifierEmail() throws Exception {
		evaluate(DECISION_TREE + "TClassifier", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeTClassifierEmailBin() throws Exception {
		evaluate(DECISION_TREE + "TClassifier", EMAIL + "Bin");
	}

	@Test
	public void evaluateDecisionTreeTRegressorEmail() throws Exception {
		evaluate(DECISION_TREE + "TRegressor", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeTRegressorEmailBin() throws Exception {
		evaluate(DECISION_TREE + "TRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateDecisionTreeXClassifierEmail() throws Exception {
		evaluate(DECISION_TREE + "XClassifier", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeXClassifierEmailBin() throws Exception {
		evaluate(DECISION_TREE + "XClassifier", EMAIL + "Bin");
	}

	@Test
	public void evaluateDecisionTreeXRegressorEmail() throws Exception {
		evaluate(DECISION_TREE + "XRegressor", EMAIL);
	}

	@Test
	public void evaluateDecisionTreeXRegressorEmailBin() throws Exception {
		evaluate(DECISION_TREE + "XRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmail() throws Exception {
		evaluate(GRADIENT_BOOSTING + "SRegressor", EMAIL);
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmailBin() throws Exception {
		evaluate(GRADIENT_BOOSTING + "SRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateGradientBoostingTRegressorEmail() throws Exception {
		evaluate(GRADIENT_BOOSTING + "TRegressor", EMAIL, new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateGradientBoostingTRegressorEmailBin() throws Exception {
		evaluate(GRADIENT_BOOSTING + "TRegressor", EMAIL + "Bin", new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateGradientBoostingXRegressorEmail() throws Exception {
		evaluate(GRADIENT_BOOSTING + "XRegressor", EMAIL, new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateGradientBoostingXRegressorEmailBin() throws Exception {
		evaluate(GRADIENT_BOOSTING + "XRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateRandomForestRClassifierEmail() throws Exception {
		evaluate(RANDOM_FOREST + "RClassifier", EMAIL);
	}

	@Test
	public void evaluateRandomForestRClassifierEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "RClassifier", EMAIL + "Bin");
	}

	@Test
	public void evaluateRandomForestRRegressorEmail() throws Exception {
		evaluate(RANDOM_FOREST + "RRegressor", EMAIL);
	}

	@Test
	public void evaluateRandomForestRRegressorEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "RRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateRandomForestSClassifierEmail() throws Exception {
		evaluate(RANDOM_FOREST + "SClassifier", EMAIL, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateRandomForestSClassifierEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "SClassifier", EMAIL + "Bin", new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateRandomForestSRegressorEmail() throws Exception {
		evaluate(RANDOM_FOREST + "SRegressor", EMAIL, new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateRandomForestSRegressorEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "SRegressor", EMAIL + "Bin", new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateRandomForestTClassifierEmail() throws Exception {
		evaluate(RANDOM_FOREST + "TClassifier", EMAIL);
	}

	@Test
	public void evaluateRandomForestTClassifierEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "TClassifier", EMAIL + "Bin");
	}

	@Test
	public void evaluateRandomForestTRegressorEmail() throws Exception {
		evaluate(RANDOM_FOREST + "TRegressor", EMAIL);
	}

	@Test
	public void evaluateRandomForestTRegressorEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "TRegressor", EMAIL + "Bin");
	}

	@Test
	public void evaluateRandomForestXClassifierEmail() throws Exception {
		evaluate(RANDOM_FOREST + "XClassifier", EMAIL);
	}

	@Test
	public void evaluateRandomForestXClassifierEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "XClassifier", EMAIL + "Bin", new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateRandomForestXRegressorEmail() throws Exception {
		evaluate(RANDOM_FOREST + "XRegressor", EMAIL);
	}

	@Test
	public void evaluateRandomForestXRegressorEmailBin() throws Exception {
		evaluate(RANDOM_FOREST + "XRegressor", EMAIL + "Bin");
	}
}