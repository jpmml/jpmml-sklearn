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

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;
import org.jpmml.converter.XMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class MultiLookupTransformer extends LookupTransformer {

	public MultiLookupTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		return super.encodeFeatures(features, encoder);
	}

	@Override
	protected List<String> formatColumns(List<Feature> features){
		List<String> result = new ArrayList<>();

		for(Feature feature : features){
			FieldName name = feature.getName();

			result.add("data:" + XMLUtil.createTagName(name.getValue()));
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
