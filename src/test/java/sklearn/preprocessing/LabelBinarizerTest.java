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
package sklearn.preprocessing;

import java.util.Arrays;

import numpy.core.NDArray;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.NormDiscrete;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class LabelBinarizerTest {

	@Test
	public void encode(){
		FieldName name = FieldName.create("x");

		NDArray array = new NDArray();
		array.put("data", Arrays.asList("low", "medium", "high"));
		array.put("fortran_order", Boolean.FALSE);

		LabelBinarizer binarizer = new LabelBinarizer("sklearn.preprocessing.label", "LabelBinarizer");
		binarizer.put("classes_", array);
		binarizer.put("pos_label", 1d);
		binarizer.put("neg_label", -1d);

		Expression expression = binarizer.encode(0, name);

		assertTrue(expression instanceof Apply);
		assertEquals("if", ((Apply)expression).getFunction());

		binarizer.put("neg_label", 0d);

		expression = binarizer.encode(0, name);

		assertTrue(expression instanceof NormDiscrete);
	}
}