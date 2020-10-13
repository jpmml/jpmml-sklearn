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
package org.jpmml.sklearn;

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.evaluator.EvaluatorBuilder;
import org.jpmml.evaluator.ModelEvaluatorBuilder;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.Batch;
import org.junit.Test;

public class BSplineTest extends SkLearnTest {

	@Override
	protected Batch createBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		Batch result = new SkLearnTestBatch(name, dataset, predicate, equivalence){

			@Override
			public BSplineTest getIntegrationTest(){
				return BSplineTest.this;
			}

			@Override
			public EvaluatorBuilder getEvaluatorBuilder() throws Exception {
				EvaluatorBuilder evaluatorBuilder = super.getEvaluatorBuilder();

				evaluatorBuilder = ((ModelEvaluatorBuilder)evaluatorBuilder)
					.setDerivedFieldGuard(null)
					.setFunctionGuard(null);

				return evaluatorBuilder;
			}
		};

		return result;
	}

	@Test
	public void evaluateGaussian() throws Exception {
		evaluate("Gaussian", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateSin() throws Exception {
		evaluate("Sin", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	@Test
	public void evaluateTanh() throws Exception {
		evaluate("Tanh", "BSpline", excludeFields(BSplineTest.predictedValue));
	}

	private static final FieldName predictedValue = FieldNameUtil.create("predict", "y");
}