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
package numpy.core;

import java.util.Arrays;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NDArrayUtilTest {

	@Test
	public void getContent(){
		String[] data = {
			"11", "12", "13",
			"21", "22", "23"
		};

		NDArray array = new NDArray();
		array.put("shape", new Object[]{2, 3});
		array.put("fortran_order", Boolean.FALSE);
		array.put("data", Arrays.asList(data));

		assertEquals(Arrays.asList(data), NDArrayUtil.getContent(array));

		array.put("fortran_order", Boolean.TRUE);

		assertEquals(Arrays.asList("11", "13", "22", "12", "21", "23"), NDArrayUtil.getContent(array));
	}
}