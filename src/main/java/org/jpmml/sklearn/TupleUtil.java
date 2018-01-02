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

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;

public class TupleUtil {

	private TupleUtil(){
	}

	static
	public Object extractElement(Object[] tuple, int i){
		return tuple[i];
	}

	static
	public <E> E extractElement(Object[] tuple, int i, Class<? extends E> clazz){
		Object value = extractElement(tuple, i);

		CastFunction<E> castFunction = new CastFunction<E>(clazz){

			@Override
			protected String formatMessage(Object object){
				return "Tuple contains an unsupported value (" + ClassDictUtil.formatClass(object) + ")";
			}
		};

		return castFunction.apply(value);
	}

	static
	public List<?> extractElementList(List<Object[]> tuples, final int i){
		Function<Object[], Object> function = new Function<Object[], Object>(){

			@Override
			public Object apply(Object[] tuple){
				return tuple[i];
			}
		};

		return Lists.transform(tuples, function);
	}

	static
	public <E> List<? extends E> extractElementList(List<Object[]> tuples, int i, Class<? extends E> clazz){
		List<?> values = extractElementList(tuples, i);

		CastFunction<E> castFunction = new CastFunction<E>(clazz){

			@Override
			protected String formatMessage(Object object){
				return "Tuple contains an unsupported value (" + ClassDictUtil.formatClass(object) +")";
			}
		};

		return Lists.transform(values, castFunction);
	}
}