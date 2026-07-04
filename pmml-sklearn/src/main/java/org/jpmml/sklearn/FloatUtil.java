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

public class FloatUtil {

	private FloatUtil(){
	}

	/**
	 * @return The largest float value that is less than or equal to the argument double value.
	 */
	static
	public float floor(double value){
		float result = (float)value;

		if((double)result > value){
			result = Math.nextDown(result);
		}

		return result;
	}

	/**
	 * @return The smallest float value that is greater than or equal to the argument double value.
	 */
	static
	public float ceil(double value){
		float result = (float)value;

		if((double)result < value){
			result = Math.nextUp(result);
		}

		return result;
	}
}