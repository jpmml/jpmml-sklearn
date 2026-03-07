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
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnAlgorithms;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class CausalMLTest extends SkLearnEncoderBatchTest implements SkLearnAlgorithms {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public CausalMLTest getArchiveBatchTest(){
				return CausalMLTest.this;
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
	public void evaluateGradientBoostingSRegressorEmail() throws Exception {
		evaluate(GRADIENT_BOOSTING + "SRegressor", "Email");
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmailBin() throws Exception {
		evaluate(GRADIENT_BOOSTING + "SRegressor", "EmailBin");
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
}