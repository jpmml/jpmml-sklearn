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
import java.util.List;

import javax.xml.parsers.DocumentBuilder;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Row;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.XMLUtil;

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
	protected List<String> createInputColumns(List<Feature> features){
		List<String> result = new ArrayList<>();

		for(Feature feature : features){
			FieldName name = feature.getName();

			result.add(XMLUtil.createTagName(name.getValue()));
		}

		return result;
	}

	@Override
	protected Row createRow(DocumentBuilder documentBuilder, List<String> columns, Object inputValue, Object outputValue){
		Object[] inputValues = (Object[])inputValue;

		ClassDictUtil.checkSize(inputValues.length + 1, columns);

		List<String> keys = new ArrayList<>();
		List<String> values = new ArrayList<>();

		for(int i = 0; i < inputValues.length; i++){

			if(inputValues[i] == null){
				continue;
			}

			keys.add(columns.get(i));
			values.add(ValueUtil.formatValue(inputValues[i]));
		}

		if(keys.isEmpty()){
			return null;
		}

		keys.add(columns.get(columns.size() - 1));
		values.add(ValueUtil.formatValue(outputValue));

		return DOMUtil.createRow(documentBuilder, keys, values);
	}
}