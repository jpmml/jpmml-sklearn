/*
 * Copyright (c) 2026 Villu Ruusmann
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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FloatUtilTest {

	@Test
	public void floor(){
		// Exact representation
		assertEquals(-2.0f, FloatUtil.floor(-2.0d));
		assertEquals(1.0f, FloatUtil.floor(1.0d));

		// Inexact representation - naive cast is wrong (returns a greater value)
		assertTrue(-1.1909019f < (float)-1.1909018158912659d);
		assertEquals(-1.1909019f, FloatUtil.floor(-1.1909018158912659d));

		// Inexact representation - naive cast is correct
		assertEquals(-1.154726f, (float)-1.154725968837738d);
		assertEquals(-1.154726f, FloatUtil.floor(-1.154725968837738d));

		// Smallest float values
		assertEquals(-Math.nextUp(Float.MIN_VALUE), FloatUtil.floor(Math.nextDown((double)-Float.MIN_VALUE)));
		assertEquals(-Float.MIN_VALUE, FloatUtil.floor((double)-Float.MIN_VALUE));
		assertEquals(-Float.MIN_VALUE, FloatUtil.floor(Math.nextUp((double)-Float.MIN_VALUE)));
		assertEquals(0f, FloatUtil.floor(Math.nextDown((double)Float.MIN_VALUE)));
		assertEquals(Float.MIN_VALUE, FloatUtil.floor((double)Float.MIN_VALUE));
		assertEquals(Float.MIN_VALUE, FloatUtil.floor(Math.nextUp((double)Float.MIN_VALUE)));

		// Smallest double values
		assertEquals(-Float.MIN_VALUE, FloatUtil.floor(-Double.MIN_VALUE));
		assertEquals(0f, FloatUtil.floor(Double.MIN_VALUE));

		// Largest float values
		assertEquals(Float.NEGATIVE_INFINITY, FloatUtil.floor(Math.nextDown((double)-Float.MAX_VALUE)));
		assertEquals(-Float.MAX_VALUE, FloatUtil.floor((double)-Float.MAX_VALUE));
		assertEquals(-Float.MAX_VALUE, FloatUtil.floor(Math.nextUp((double)-Float.MAX_VALUE)));
		assertEquals(Math.nextDown(Float.MAX_VALUE), FloatUtil.floor(Math.nextDown((double)Float.MAX_VALUE)));
		assertEquals(Float.MAX_VALUE, FloatUtil.floor((double)Float.MAX_VALUE));
		assertEquals(Float.MAX_VALUE, FloatUtil.floor(Math.nextUp((double)Float.MAX_VALUE)));

		// Largest double values
		assertEquals(Float.NEGATIVE_INFINITY, FloatUtil.floor(-Double.MAX_VALUE));
		assertEquals(Float.MAX_VALUE, FloatUtil.floor(Double.MAX_VALUE));
	}

	@Test
	public void ceil(){
		// Exact representation
		assertEquals(-2.0f, FloatUtil.ceil(-2.0d));
		assertEquals(1.0f, FloatUtil.ceil(1.0d));

		// Inexact representation - naive cast is correct
		assertEquals(-1.1909018f, (float)-1.1909018158912659d);
		assertEquals(-1.1909018f, FloatUtil.ceil(-1.1909018158912659d));

		// Inexact representation - naive cast is wrong (returns a lesser value)
		assertTrue(-1.1547259f > (float)-1.154725968837738d);
		assertEquals(-1.1547259f, FloatUtil.ceil(-1.154725968837738d));

		// Smallest float values
		assertEquals(-Float.MIN_VALUE, FloatUtil.ceil(Math.nextDown((double)-Float.MIN_VALUE)));
		assertEquals(-Float.MIN_VALUE, FloatUtil.ceil((double)-Float.MIN_VALUE));
		assertEquals(-0f, FloatUtil.ceil(Math.nextUp((double)-Float.MIN_VALUE)));
		assertEquals(Float.MIN_VALUE, FloatUtil.ceil(Math.nextDown((double)Float.MIN_VALUE)));
		assertEquals(Float.MIN_VALUE, FloatUtil.ceil((double)Float.MIN_VALUE));
		assertEquals(Math.nextUp(Float.MIN_VALUE), FloatUtil.ceil(Math.nextUp((double)Float.MIN_VALUE)));

		// Smallest double values
		assertEquals(-0f, FloatUtil.ceil(-Double.MIN_VALUE));
		assertEquals(Float.MIN_VALUE, FloatUtil.ceil(Double.MIN_VALUE));

		// Largest float values
		assertEquals(-Float.MAX_VALUE, FloatUtil.ceil(Math.nextDown((double)-Float.MAX_VALUE)));
		assertEquals(-Float.MAX_VALUE, FloatUtil.ceil((double)-Float.MAX_VALUE));
		assertEquals(-Math.nextDown(Float.MAX_VALUE), FloatUtil.ceil(Math.nextUp((double)-Float.MAX_VALUE)));
		assertEquals(Float.MAX_VALUE, FloatUtil.ceil(Math.nextDown((double)Float.MAX_VALUE)));
		assertEquals(Float.MAX_VALUE, FloatUtil.ceil((double)Float.MAX_VALUE));
		assertEquals(Float.POSITIVE_INFINITY, FloatUtil.ceil(Math.nextUp((double)Float.MAX_VALUE)));

		// Largest double values
		assertEquals(-Float.MAX_VALUE, FloatUtil.ceil(-Double.MAX_VALUE));
		assertEquals(Float.POSITIVE_INFINITY, FloatUtil.ceil(Double.MAX_VALUE));
	}

	@Test
	public void duality(){
		checkDuality(-2.0d);
		checkDuality(-0d);
		checkDuality(0d);
		checkDuality(1.0d);

		checkDuality(-1.1909018158912659d);
		checkDuality(-1.154725968837738d);
	}

	static
	private void checkDuality(double value){
		assertEquals(FloatUtil.ceil(value), -FloatUtil.floor(-value), 0f);
		assertEquals(FloatUtil.floor(value), -FloatUtil.ceil(-value), 0f);
	}
}