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
import org.junit.Test;
import sklearn.Estimator;

public class BSplineTest extends SkLearnEncoderBatchTest {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public BSplineTest getArchiveBatchTest(){
				return BSplineTest.this;
			}

			@Override
			public String getSeparator(){
				return "\t";
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
	public void evaluateCubic2() throws Exception {
		evaluate("Cubic2", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateCubic3() throws Exception {
		evaluate("Cubic3", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateGaussian() throws Exception {
		evaluate("Gaussian", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic2() throws Exception {
		evaluate("Quadratic2", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic3() throws Exception {
		evaluate("Quadratic3", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic4() throws Exception {
		evaluate("Quadratic4", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateQuadratic5() throws Exception {
		evaluate("Quadratic5", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateSin() throws Exception {
		evaluate("Sin", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateTanh() throws Exception {
		evaluate("Tanh", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	private static final String predictedValue = FieldNameUtil.create(Estimator.FIELD_PREDICT, "y");
}