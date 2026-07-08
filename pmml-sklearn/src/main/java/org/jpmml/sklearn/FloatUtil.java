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

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.dmg.pmml.HasContinuousDomain;
import org.dmg.pmml.HasDiscreteDomain;
import org.dmg.pmml.Interval;

public class FloatUtil {

	private FloatUtil(){
	}

	static
	public <E extends Field<?>> E narrow(E field){
		field.setDataType(DataType.FLOAT);

		if(field instanceof HasDiscreteDomain){
			HasDiscreteDomain<?> hasDiscreteDomain = (HasDiscreteDomain<?>)field;

			// Ignored
		} // End if

		if(field instanceof HasContinuousDomain){
			HasContinuousDomain<?> hasContinuousDomain = (HasContinuousDomain<?>)field;

			if(hasContinuousDomain.hasIntervals()){
				List<Interval> intervals = hasContinuousDomain.getIntervals();

				for(Interval interval : intervals){
					Number leftMargin = interval.getLeftMargin();
					Number rightMargin = interval.getRightMargin();

					if(leftMargin != null && !(leftMargin instanceof Float)){
						interval.setLeftMargin(leftMargin.floatValue());
					} // End if

					if(rightMargin != null && !(rightMargin instanceof Float)){
						interval.setRightMargin(rightMargin.floatValue());
					}
				}
			}
		}

		return field;
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