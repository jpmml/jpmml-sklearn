/*
 * Copyright (c) 2022 Villu Ruusmann
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
package chaid;

import java.util.List;

import com.google.common.collect.Lists;
import numpy.core.ScalarUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.PythonObject;

public class Split extends PythonObject {

	public Split(String module, String name){
		super(module, name);
	}

	public Integer getColumnId(){
		Object columnId = get("column_id");

		if(columnId == null){
			return null;
		}

		return getInteger("column_id");
	}

	public InvalidSplitReason getInvalidReason(){
		return getOptional("_invalid_reason", InvalidSplitReason.class);
	}

	public List<List<Integer>> getSplits(){
		List<?> splits = getList("splits");

		return decodeSplits(splits);
	}

	public List<List<?>> getSplitMap(){
		List<?> splitMap = getList("split_map");

		return decodeSplitMap(splitMap);
	}

	static
	private List<List<Integer>> decodeSplits(List<?> splits){
		return Lists.transform(splits, split -> {
			return Lists.transform((List<?>)split, value -> {
				return ValueUtil.asInteger((Number)ScalarUtil.decode(value));
			});
		});
	}

	static
	public List<List<?>> decodeSplitMap(List<?> splits){
		return Lists.transform(splits, split -> {
			return Lists.transform((List<?>)split, value -> {
				return ScalarUtil.decode(value);
			});
		});
	}
}