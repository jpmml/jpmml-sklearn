/*
 * Copyright (c) 2022 Villu Ruusmann
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

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;

public class PyCaretTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluatePyCaretAudit() throws Exception {
		evaluate("PyCaret", AUDIT);
	}

	@Test
	public void evaluatePyCaretAuto() throws Exception {
		evaluate("PyCaret", AUTO, new FloatEquivalence(1));
	}

	@Test
	public void evaluatePyCaretIris() throws Exception {
		evaluate("PyCaret", IRIS);
	}

	@Test
	public void evaluatePyCaretWheat() throws Exception {
		evaluate("PyCaret", WHEAT, excludeFields(createAffinityFields(4)));
	}

	static
	private String[] createAffinityFields(int count){
		List<OutputField> affinityFields = new ArrayList<>();

		for(int i = 0; i < count; i++){
			affinityFields.add(ModelUtil.createAffinityField(DataType.DOUBLE, i));
		}

		return affinityFields.stream()
			.map(OutputField::requireName)
			.toArray(String[]::new);
	}
}