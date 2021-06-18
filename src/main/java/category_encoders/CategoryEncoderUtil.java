/*
 * Copyright (c) 2021 Villu Ruusmann
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
package category_encoders;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.common.base.Function;

public class CategoryEncoderUtil {

	private CategoryEncoderUtil(){
	}

	static
	public <K, V> Map<K, V> toTransformedMap(Map<?, ?> map, Function<Object, K> keyFunction, Function<Object, V> valueFunction){
		Map<K, V> result = new LinkedHashMap<>();

		Collection<? extends Map.Entry<?, ?>> entries = map.entrySet();
		for(Map.Entry<?, ?> entry : entries){
			K key = keyFunction.apply(entry.getKey());
			V value = valueFunction.apply(entry.getValue());

			result.put(key, value);
		}

		return result;
	}
}