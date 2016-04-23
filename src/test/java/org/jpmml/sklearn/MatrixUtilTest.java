/*
 * Copyright (c) 2016 Villu Ruusmann
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

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MatrixUtilTest {

	@Test
	public void getRow(){
		List<String> values = Arrays.asList(MatrixUtilTest.DATA);

		assertEquals(Arrays.asList("11", "12", "13", "14"), MatrixUtil.getRow(values, 3, 4, 0));
		assertEquals(Arrays.asList("21", "22", "23", "24"), MatrixUtil.getRow(values, 3, 4, 1));
		assertEquals(Arrays.asList("31", "32", "33", "34"), MatrixUtil.getRow(values, 3, 4, 2));
	}

	@Test
	public void getColumn(){
		List<String> values = Arrays.asList(MatrixUtilTest.DATA);

		assertEquals(Arrays.asList("11", "21", "31"), MatrixUtil.getColumn(values, 3, 4, 0));
		assertEquals(Arrays.asList("12", "22", "32"), MatrixUtil.getColumn(values, 3, 4, 1));
		assertEquals(Arrays.asList("13", "23", "33"), MatrixUtil.getColumn(values, 3, 4, 2));
		assertEquals(Arrays.asList("14", "24", "34"), MatrixUtil.getColumn(values, 3, 4, 3));
	}

	private static final String[] DATA = new String[]{
		"11", "12", "13", "14",
		"21", "22", "23", "24",
		"31", "32", "33", "34"
	};
}