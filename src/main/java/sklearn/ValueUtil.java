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

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.google.common.math.DoubleMath;
import com.google.common.primitives.Ints;

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

	static
	public List<Integer> asIntegers(List<? extends Number> numbers){

		if(numbers == null){
			return null;
		}

		Function<Number, Integer> function = new Function<Number, Integer>(){

			@Override
			public Integer apply(Number number){

				if(number instanceof Integer){
					return (Integer)number;
				}

				double value = number.doubleValue();

				if(DoubleMath.isMathematicalInteger(value)){
					return Ints.checkedCast((long)value);
				}

				throw new IllegalArgumentException();
			}
		};

		return Lists.transform(numbers, function);
	}

	static
	public List<Double> asDoubles(List<? extends Number> numbers){

		if(numbers == null){
			return null;
		}

		Function<Number, Double> function = new Function<Number, Double>(){

			@Override
			public Double apply(Number number){

				if(number instanceof Double){
					return (Double)number;
				}

				return Double.valueOf(number.doubleValue());
			}
		};

		return Lists.transform(numbers, function);
	}

	private static final double TOLERANCE = 1e-15;
}