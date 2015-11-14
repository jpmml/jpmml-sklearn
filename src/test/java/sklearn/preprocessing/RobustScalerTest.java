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

import org.dmg.pmml.Apply;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class RobustScalerTest {

	@Test
	public void encode(){
		FieldName name = FieldName.create("x");

		RobustScaler scaler = new RobustScaler("sklearn.preprocessing.data", "RobustScaler");
		scaler.put("with_centering", Boolean.FALSE);
		scaler.put("with_scaling", Boolean.FALSE);
		scaler.put("center_", 6);
		scaler.put("scale_", 2);

		Expression expression = scaler.encode(0, name);

		assertTrue(expression instanceof FieldRef);

		scaler.put("with_centering", Boolean.TRUE);
		scaler.put("with_scaling", Boolean.TRUE);

		expression = scaler.encode(0, name);

		assertTrue(expression instanceof Apply);
		assertEquals("/", ((Apply)expression).getFunction());

		scaler.put("scale_", 1);

		expression = scaler.encode(0, name);

		assertTrue(expression instanceof Apply);
		assertEquals("-", ((Apply)expression).getFunction());

		scaler.put("center_", 0);

		expression = scaler.encode(0, name);

		assertTrue(expression instanceof FieldRef);
	}
}