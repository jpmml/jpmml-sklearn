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
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class CausalMLTest extends SkLearnEncoderBatchTest {

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
	public void evaluateDecisionTreeSRegressorEmail() throws Exception {
		evaluate("DecisionTreeSRegressor", "Email");
	}

	@Test
	public void evaluateDecisionTreeSRegressorEmailBin() throws Exception {
		evaluate("DecisionTreeSRegressor", "EmailBin");
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmail() throws Exception {
		evaluate("GradientBoostingSRegressor", "Email");
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmailBin() throws Exception {
		evaluate("GradientBoostingSRegressor", "EmailBin");
	}

	@Test
	public void evaluateRandomForestSRegressorEmail() throws Exception {
		evaluate("RandomForestSRegressor", "Email", new PMMLEquivalence(5e-11, 5e-11));
	}

	@Test
	public void evaluateRandomForestSRegressorEmailBin() throws Exception {
		evaluate("RandomForestSRegressor", "EmailBin", new PMMLEquivalence(5e-11, 5e-11));
	}
}