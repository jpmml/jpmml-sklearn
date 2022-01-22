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

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ResultField;

abstract
public class MarkupTest extends SkLearnEncoderBatchTest {

	abstract
	public void check(PMML pmml);

	public void check(String name, String dataset) throws Exception {
		Predicate<ResultField> predicate = (resultField) -> true;
		Equivalence<Object> equivalence = getEquivalence();

		try(SkLearnEncoderBatch batch = (SkLearnEncoderBatch)createBatch(name, dataset, predicate, equivalence)){
			PMML pmml = batch.getPMML();

			check(pmml);
		}
	}
}