/*
 * Copyright (c) 2018 Villu Ruusmann
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

import com.google.common.base.Function;

abstract
public class CastFunction<E> implements Function<Object, E> {

	private Class<? extends E> clazz = null;


	public CastFunction(Class<? extends E> clazz){
		setClazz(clazz);
	}

	abstract
	protected String formatMessage(Object object);

	@Override
	public E apply(Object object){
		Class<? extends E> clazz = getClazz();

		try {
			return clazz.cast(object);
		} catch(ClassCastException cce){
			throw new IllegalArgumentException(formatMessage(object), cce);
		}
	}

	public Class<? extends E> getClazz(){
		return this.clazz;
	}

	private void setClazz(Class<? extends E> clazz){

		if(clazz == null){
			throw new IllegalArgumentException();
		}

		this.clazz = clazz;
	}
}