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
package sklearn;

import com.google.common.math.DoubleMath;

public class ValueUtil {

	private ValueUtil(){
	}

	static
	public boolean isZero(Number number){
		return DoubleMath.fuzzyEquals(number.doubleValue(), 0d, ValueUtil.TOLERANCE);
	}

	static
	public boolean isOne(Number number){
		return DoubleMath.fuzzyEquals(number.doubleValue(), 1d, ValueUtil.TOLERANCE);
	}

	private static final double TOLERANCE = 1e-15;
}