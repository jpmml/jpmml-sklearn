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
package numpy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.dmg.pmml.DataType;
import org.jpmml.sklearn.CClassDict;

public class DType extends CClassDict {

	public DType(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){
		super.__setstate__(createAttributeMap(INIT_ATTRIBUTES, args));
	}

	/**
	 * https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/descriptor.c
	 */
	@Override
	public void __setstate__(Object[] args){
		super.__setstate__(createAttributeMap(SETSTATE_ATTRIBUTES, args));
	}

	public DataType getDataType(){
		String className = getClassName();

		switch(className){
			case "numpy.bool_":
				return DataType.BOOLEAN;
			case "numpy.int_":
			case "numpy.int8":
			case "numpy.int16":
			case "numpy.int32":
			case "numpy.int64":
				return DataType.INTEGER;
			case "numpy.float32":
				return DataType.FLOAT;
			case "numpy.float_":
			case "numpy.float64":
				return DataType.DOUBLE;
			default:
				throw new IllegalArgumentException(className);
		}
	}

	public Object toDescr(){
		Map<String, Object[]> values = getValues();

		if(values == null){
			String obj = getObj();
			String order = getOrder();

			return formatDescr(obj, order);
		}

		Set<String> valueKeys = values.keySet();

		if((TREE_KEYS).equals(valueKeys)){
			return formatDescr(TREE_KEYS, values);
		} else

		if((NODEDATA_KEYS).equals(valueKeys)){
			return formatDescr(NODEDATA_KEYS, values);
		}

		throw new IllegalArgumentException();
	}

	public Map<String, Object[]> getValues(){
		return (Map)get("values");
	}

	public String getObj(){
		return (String)get("obj");
	}

	public String getOrder(){
		return (String)get("order");
	}

	static
	private List<Object[]> formatDescr(Collection<String> keys, Map<String, Object[]> values){
		List<Object[]> result = new ArrayList<>();

		for(String key : keys){
			Object[] value = values.get(key);

			DType dType = (DType)value[0];

			result.add(new Object[]{key, dType.toDescr()});
		}

		return result;
	}

	static
	private String formatDescr(String obj, String order){

		if(obj == null){
			throw new IllegalArgumentException();
		}

		return (order != null ? (order + obj) : obj);
	}

	private static final String[] INIT_ATTRIBUTES = {
		"obj",
		"align",
		"copy"
	};

	private static final String[] SETSTATE_ATTRIBUTES = {
		"version",
		"order",
		"subdescr",
		"names",
		"values",
		"w_size",
		"alignment",
		"flags"
	};

	private static final Set<String> TREE_KEYS = new LinkedHashSet<>(Arrays.asList("left_child", "right_child", "feature", "threshold", "impurity", "n_node_samples", "weighted_n_node_samples"));
	private static final Set<String> NODEDATA_KEYS = new LinkedHashSet<>(Arrays.asList("idx_start", "idx_end", "is_leaf", "radius"));
}