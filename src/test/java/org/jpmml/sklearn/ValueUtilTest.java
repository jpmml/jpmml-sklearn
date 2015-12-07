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
package org.jpmml.sklearn;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class ValueUtilTest {

	@Test
	public void asInteger(){
		assertEquals((Integer)0, ValueUtil.asInteger((Integer)0));
		assertEquals((Integer)0, ValueUtil.asInteger((Long)0L));
		assertEquals((Integer)0, ValueUtil.asInteger((Float)0f));
		assertEquals((Integer)0, ValueUtil.asInteger((Double)0d));

		try {
			ValueUtil.asInteger((Double)0.5d);

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}

		try {
			ValueUtil.asInteger((Double)(Integer.MAX_VALUE + 0.5d));

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}
	}
}