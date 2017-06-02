/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn;

import java.util.Arrays;

import org.dmg.pmml.DataType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TypeUtilTest {

	@Test
	public void getDataType(){
		assertEquals(DataType.INTEGER, TypeUtil.getDataType(Arrays.asList(0, "1", 2), DataType.STRING));
		assertEquals(DataType.FLOAT, TypeUtil.getDataType(Arrays.asList(0f, "1.0", 2f), DataType.STRING));
		assertEquals(DataType.DOUBLE, TypeUtil.getDataType(Arrays.asList(0d, "1.0", 2d), DataType.STRING));

		try {
			TypeUtil.getDataType(Arrays.asList(0, 1f, 2d), DataType.STRING);

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}
	}
}