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

import java.util.List;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.ModelUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.python.testing.PythonEncoderBatchTest;

abstract
public class SkLearnEncoderBatchTest extends PythonEncoderBatchTest {

	public SkLearnEncoderBatchTest(){
		this(new PMMLEquivalence(1e-13, 1e-13));
	}

	public SkLearnEncoderBatchTest(Equivalence<Object> equivalence){
		super(equivalence);
	}

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public SkLearnEncoderBatchTest getArchiveBatchTest(){
				return SkLearnEncoderBatchTest.this;
			}
		};

		return result;
	}

	static
	protected String[] createNeighborFields(int count){
		List<OutputField> neighborFields = ModelUtil.createNeighborFields(count);

		return neighborFields.stream()
			.map(OutputField::requireName)
			.toArray(String[]::new);
	}
}