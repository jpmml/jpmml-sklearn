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
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.jpmml.model.visitors.VisitorBattery;
import org.junit.Test;

public class ClustererTest extends ValidatingSkLearnEncoderBatchTest implements SkLearnAlgorithms, Datasets {

	@Override
	public ValidatingSkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		ValidatingSkLearnEncoderBatch result = new ValidatingSkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public ClustererTest getArchiveBatchTest(){
				return ClustererTest.this;
			}

			@Override
			public VisitorBattery getValidators(){
				VisitorBattery visitorBattery = super.getValidators();

				visitorBattery.add(ModelStatsInspector.class);

				return visitorBattery;
			}
		};

		return result;
	}

	@Test
	public void evaluateKMeansWheat() throws Exception {
		evaluate(K_MEANS, WHEAT);
	}

	@Test
	public void evaluateMiniBatchKMeansWheat() throws Exception {
		evaluate(MINIBATCH_K_MEANS, WHEAT);
	}
}