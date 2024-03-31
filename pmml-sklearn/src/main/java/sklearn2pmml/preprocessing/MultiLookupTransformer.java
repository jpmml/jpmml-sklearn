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
package sklearn2pmml.preprocessing;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.dmg.pmml.DataType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.XMLUtil;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasMultiType;

public class MultiLookupTransformer extends LookupTransformer implements HasMultiType {

	public MultiLookupTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public DataType getDataType(int index){
		Map<?, ?> mapping = getMapping();

		List<?> values = (mapping.entrySet()).stream()
			.map(entry -> {
				return TupleUtil.extractElement((Object[])entry.getKey(), index);
			})
			.collect(Collectors.toList());

		return TypeUtil.getDataType(values, DataType.STRING);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		return super.encodeFeatures(features, encoder);
	}

	@Override
	protected List<String> formatColumns(List<Feature> features){
		List<String> result = new ArrayList<>();

		for(Feature feature : features){
			String name = feature.getName();

			result.add("data:" + XMLUtil.createTagName(name));
		}

		if(result.contains("data:output")){
			throw new IllegalArgumentException();
		}

		result.add("data:output");

		return result;
	}

	@Override
	protected Map<String, List<Object>> parseMapping(List<String> inputColumns, String outputColumn, Map<?, ?> mapping){
		Map<String, List<Object>> result = new LinkedHashMap<>();

		for(String inputColumn : inputColumns){
			result.put(inputColumn, new ArrayList<>());
		}

		result.put(outputColumn, new ArrayList<>());

		Collection<? extends Map.Entry<?, ?>> entries = mapping.entrySet();
		for(Map.Entry<?, ?> entry : entries){
			Object[] inputValue = (Object[])entry.getKey();
			Object outputValue = entry.getValue();

			if(inputValue == null || inputValue.length != inputColumns.size()){
				throw new IllegalArgumentException();
			} // End if

			if(outputValue == null){
				continue;
			}

			for(int i = 0; i < inputColumns.size(); i++){
				List<Object> inputValues = result.get(inputColumns.get(i));

				inputValues.add(inputValue[i]);
			}

			List<Object> outputValues = result.get(outputColumn);

			outputValues.add(outputValue);
		}

		return result;
	}
}
