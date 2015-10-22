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

public class StandardScalerTest {

	@Test
	public void encode(){
		FieldName name = FieldName.create("x");

		StandardScaler scaler = new StandardScaler("sklearn.preprocessing.data", "StandardScaler");
		scaler.put("with_mean", Boolean.TRUE);
		scaler.put("with_std", Boolean.TRUE);
		scaler.put("mean_", 6d);
		scaler.put("std_", 2d);

		Expression expression = scaler.encode(name);

		assertTrue(expression instanceof Apply);
		assertEquals("/", ((Apply)expression).getFunction());

		scaler.put("std_", 1);

		expression = scaler.encode(name);

		assertTrue(expression instanceof Apply);
		assertEquals("-", ((Apply)expression).getFunction());

		scaler.put("mean_", 0);

		expression = scaler.encode(name);

		assertTrue(expression instanceof FieldRef);
	}
}