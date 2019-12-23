/*
 * Copyright (c) 2019 Villu Ruusmann
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

public class CastUtil {

	private CastUtil(){
	}

	static
	public Object castTo(Object object, Class<?> clazz){

		if(object instanceof Castable){
			Castable castable = (Castable)object;

			return castable.castTo(clazz);
		}

		return object;
	}

	static
	public Object deepCastTo(Object object, Class<?> clazz){

		while(true){
			Object castObject = castTo(object, clazz);

			if(castObject == object){
				break;
			}

			object = castObject;
		}

		return object;
	}
}