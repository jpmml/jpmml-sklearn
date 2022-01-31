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

import java.util.Map;

import numpy.core.ScalarUtil;
import pandas.core.Series;

abstract
public class MapEncoder extends CategoryEncoder {

	public MapEncoder(String module, String name){
		super(module, name);
	}

	abstract
	public String functionName();

	public Map<Object, Series> getMapping(){
		Map<?, ?> mapping = get("mapping", Map.class);

		return CategoryEncoderUtil.toTransformedMap(mapping, key -> ScalarUtil.decode(key), value -> (Series)value);
	}
}