/*
 * Copyright (c) 2021 Villu Ruusmann
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

import java.io.IOException;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.FieldNames;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.junit.jupiter.api.Test;
import sklearn.Estimator;
import sklearn.SkLearnMethods;

public class SkLearn2PMMLTest extends SkLearnEncoderBatchTest implements SkLearnAlgorithms, Datasets, Fields {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public SkLearnEncoderBatchTest getArchiveBatchTest(){
				return SkLearn2PMMLTest.this;
			}

			@Override
			public Table getInput() throws IOException {
				Table table = super.getInput();

				String algorithm = getAlgorithm();
				String dataset = getDataset();

				if(Objects.equals(LINEAR_REGRESSION, algorithm) && Objects.equals("Airline", dataset)){
					Function<Object, Number> function = new Function<Object, Number>(){

						@Override
						public Number apply(Object value){
							String string = (String)value;

							return Integer.valueOf(string);
						}
					};

					table.apply("Passengers", function);
				}

				return table;
			}
		};

		return result;
	}

	@Test
	public void evaluateDayOfWeekApollo() throws Exception {
		evaluate("DayMonthYear", "Apollo");
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
	public void evaluateLinearRegressionAirline() throws Exception {
		evaluate(LINEAR_REGRESSION, "Airline");
	}

	@Test
	public void evaluateCHAIDAudit() throws Exception {
		evaluate("CHAID", AUDIT, excludeFields(AUDIT_ADJUSTED, AUDIT_PROBABILITY_FALSE, AUDIT_PROBABILITY_TRUE));
	}

	@Test
	public void evaluateGBDTLRAudit() throws Exception {
		evaluate("GBDTLR", AUDIT);
	}

	@Test
	public void evaluateMultiEstimatorChainAudit() throws Exception {
		String[] resultFields = {
			FieldNameUtil.create(FieldNames.PROBABILITY, "Male"), FieldNameUtil.create(FieldNames.PROBABILITY, "Female"),
			FieldNames.NODE_ID,
			FieldNameUtil.create(FieldNames.PROBABILITY, "Adjusted", 0), FieldNameUtil.create(FieldNames.PROBABILITY, "Adjusted", 1)
		};

		evaluate("MultiEstimatorChain", AUDIT, excludeFields(resultFields));
	}

	@Test
	public void evaluateCHAIDAuditNA() throws Exception {
		evaluate("CHAID", AUDIT_NA, excludeFields(AUDIT_ADJUSTED, AUDIT_PROBABILITY_FALSE, AUDIT_PROBABILITY_TRUE));
	}

	@Test
	public void evaluateCHAIDAuto() throws Exception {
		evaluate("CHAID", AUTO, excludeFields(AUTO_MPG));
	}

	@Test
	public void evaluateExpressionAuto() throws Exception {
		evaluate("Expression", AUTO);
	}

	@Test
	public void evaluateGBDTLMAuto() throws Exception {
		evaluate("GBDTLM", AUTO);
	}

	@Test
	public void evaluateMultiEstimatorChainAuto() throws Exception {
		evaluate("MultiEstimatorChain", AUTO, excludeFields(FieldNameUtil.create(Estimator.FIELD_PREDICT, "acceleration"), FieldNames.NODE_ID));
	}

	@Test
	public void evaluateCHAIDAutoNA() throws Exception {
		evaluate("CHAID", AUTO_NA, excludeFields(AUTO_MPG));
	}

	@Test
	public void evaluateGBDTLMHousing() throws Exception {
		evaluate("GBDTLM", HOUSING);
	}

	@Test
	public void evaluateCHAIDIris() throws Exception {
		evaluate("CHAID", IRIS, excludeFields(IRIS_SPECIES, IRIS_PROBABILITY_SETOSA, IRIS_PROBABILITY_VERSICOLOR, IRIS_PROBABILITY_VIRGINICA));
	}

	@Test
	public void evaluateExpressionIris() throws Exception {
		evaluate("Expression", IRIS);
	}

	@Test
	public void evaluateMLPAutoencoderIris() throws Exception {
		evaluate("MLPAutoencoder", IRIS);
	}

	@Test
	public void evaluateMLPTransformerIris() throws Exception {
		evaluate("MLPTransformer", IRIS);
	}

	@Test
	public void evaluatePCAIris() throws Exception {
		evaluate("PCA", IRIS, excludeFields(IRIS_PROBABILITY_SETOSA, IRIS_PROBABILITY_VERSICOLOR, IRIS_PROBABILITY_VIRGINICA, FieldNameUtil.create(SkLearnMethods.PREDICT, IRIS_SPECIES)));
	}

	@Test
	public void evaluateRuleSetIris() throws Exception {
		evaluate("RuleSet", IRIS);
	}

	@Test
	public void evaluateSelectFirstIris() throws Exception {
		evaluate("SelectFirst", IRIS, excludeFields(IRIS_PROBABILITY_SETOSA, IRIS_PROBABILITY_VERSICOLOR, IRIS_PROBABILITY_VIRGINICA));
	}

	@Test
	public void evaluateExpressionVersicolor() throws Exception {
		evaluate("Expression", VERSICOLOR);
	}

	@Test
	public void evaluateGBDTLRVersicolor() throws Exception {
		evaluate("GBDTLR", VERSICOLOR);
	}

	@Test
	public void evaluateEstimatorChainWine() throws Exception {
		evaluate("EstimatorChain", WINE);
	}

	@Test
	public void evaluateSelectFirstWine() throws Exception {
		evaluate("SelectFirst", WINE);
	}
}