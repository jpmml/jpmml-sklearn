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
package sklearn;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;

public class TypeUtil {

	private TypeUtil(){
	}

	static
	public DataType getDataType(List<?> objects, DataType defaultDataType){
		Function<Object, Class<?>> function = new Function<Object, Class<?>>(){

			@Override
			public Class<?> apply(Object object){
				return object.getClass();
			}
		};

		Set<Class<?>> clazzes = new HashSet<>(Lists.transform(objects, function));

		// A String value can be parsed to any other value
		if(clazzes.size() > 1){
			clazzes.remove(String.class);
		}

		Class<?> clazz = Iterables.getOnlyElement(clazzes);

		DataType dataType = TypeUtil.dataTypes.get(clazz);
		if(dataType != null){
			return dataType;
		}

		return defaultDataType;
	}

	private static final Map<Class<?>, DataType> dataTypes = new LinkedHashMap<>();

	static {
		dataTypes.put(Boolean.class, DataType.BOOLEAN);
		dataTypes.put(Byte.class, DataType.INTEGER);
		dataTypes.put(Short.class, DataType.INTEGER);
		dataTypes.put(Integer.class, DataType.INTEGER);
		dataTypes.put(Long.class, DataType.INTEGER);
		dataTypes.put(Float.class, DataType.FLOAT);
		dataTypes.put(Double.class, DataType.DOUBLE);
	}
}