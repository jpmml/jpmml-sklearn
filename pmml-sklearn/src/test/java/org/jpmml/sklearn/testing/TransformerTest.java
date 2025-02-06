/*
 * Copyright (c) 2020 Villu Ruusmann
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
import org.jpmml.evaluator.EvaluatorBuilder;
import org.jpmml.evaluator.ModelEvaluatorBuilder;
import org.jpmml.evaluator.ResultField;
import org.junit.jupiter.api.Test;
import sklearn.Estimator;

public class TransformerTest extends SkLearnEncoderBatchTest {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public TransformerTest getArchiveBatchTest(){
				return TransformerTest.this;
			}

			@Override
			public char getSeparator(){
				return '\t';
			}

			@Override
			public EvaluatorBuilder getEvaluatorBuilder() throws Exception {
				ModelEvaluatorBuilder evaluatorBuilder = (ModelEvaluatorBuilder)super.getEvaluatorBuilder();

				evaluatorBuilder
					.setDerivedFieldGuard(null)
					.setFunctionGuard(null);

				return evaluatorBuilder;
			}
		};

		return result;
	}

	@Test
	public void evaluateCubic2Spline() throws Exception {
		evaluate("Cubic2", "Spline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateCubic3Spline() throws Exception {
		evaluate("Cubic3", "Spline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic2Spline() throws Exception {
		evaluate("Quadratic2", "Spline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic3Spline() throws Exception {
		evaluate("Quadratic3", "Spline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic4Spline() throws Exception {
		evaluate("Quadratic4", "Spline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic5Spline() throws Exception {
		evaluate("Quadratic5", "Spline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateSimpleBoxCox() throws Exception {
		evaluate("Plain", "BoxCox", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateStandardizedBoxCox() throws Exception {
		evaluate("Standardized", "BoxCox", excludeFields(TransformerTest.predictedValue, FieldNameUtil.create("power", TransformerTest.predictedValue)));
	}

	@Test
	public void evaluateSimpleYeoJohnson() throws Exception {
		evaluate("Plain", "YeoJohnson", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateStandardizedYeoJohnson() throws Exception {
		evaluate("Standardized", "YeoJohnson", excludeFields(TransformerTest.predictedValue, FieldNameUtil.create("power", TransformerTest.predictedValue)));
	}

	@Test
	public void evaluateGaussianBSpline() throws Exception {
		evaluate("Gaussian", "BSpline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateSinBSpline() throws Exception {
		evaluate("Sin", "BSpline", excludeFields(TransformerTest.predictedValue));
	}

	@Test
	public void evaluateTanhBSpline() throws Exception {
		evaluate("Tanh", "BSpline", excludeFields(TransformerTest.predictedValue));
	}

	private static final String predictedValue = FieldNameUtil.create(Estimator.FIELD_PREDICT, "y");
}