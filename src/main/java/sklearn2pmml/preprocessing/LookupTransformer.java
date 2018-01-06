/*
 * Copyright (c) 2017 Villu Ruusmann
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
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Row;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn.TypeUtil;

public class LookupTransformer extends Transformer {

	public LookupTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		Map<?, ?> mapping = getMapping();

		List<Object> inputValues = new ArrayList<>(mapping.keySet());

		return TypeUtil.getDataType(inputValues, DataType.STRING);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Map<?, ?> mapping = getMapping();
		Object defaultValue = getDefaultValue();

		List<String> columns = new ArrayList<>();
		columns.addAll(createInputColumns(features));

		if(columns.contains("output")){
			throw new IllegalArgumentException();
		}

		columns.add("output");

		List<Object> outputValues = new ArrayList<>(mapping.size() + 1);

		DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

		InlineTable inlineTable = new InlineTable();

		Collection<? extends Map.Entry<?, ?>> entries = mapping.entrySet();
		for(Map.Entry<?, ?> entry : entries){
			Object inputValue = entry.getKey();
			Object outputValue = entry.getValue();

			if(outputValue == null){
				continue;
			}

			Row row = createRow(documentBuilder, columns, inputValue, outputValue);
			if(row == null){
				continue;
			}

			inlineTable.addRows(row);

			outputValues.add(outputValue);
		}

		StringBuilder sb = new StringBuilder();

		MapValues mapValues = new MapValues()
			.setInlineTable(inlineTable);

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			String column = columns.get(i);

			if(i > 0){
				sb.append(", ");
			}

			sb.append((FeatureUtil.getName(feature)).getValue());

			mapValues.addFieldColumnPairs(new FieldColumnPair(feature.getName(), column));
		}

		mapValues.setOutputColumn(columns.get(columns.size() - 1));

		if(defaultValue != null){
			mapValues.setDefaultValue(ValueUtil.formatValue(defaultValue));

			outputValues.add(defaultValue);
		}

		DerivedField derivedField = encoder.createDerivedField(FieldName.create("lookup(" + sb.toString() + ")"), OpType.CATEGORICAL, TypeUtil.getDataType(outputValues, DataType.STRING), mapValues);

		Feature feature = new Feature(encoder, derivedField.getName(), derivedField.getDataType()){

			@Override
			public ContinuousFeature toContinuousFeature(){
				PMMLEncoder encoder = getEncoder();

				DerivedField derivedField = (DerivedField)encoder.toContinuous(getName());

				return new ContinuousFeature(encoder, derivedField);
			}
		};

		return Collections.singletonList(feature);
	}

	protected List<String> createInputColumns(List<Feature> features){
		ClassDictUtil.checkSize(1, features);

		return Collections.singletonList("input");
	}

	protected Row createRow(DocumentBuilder documentBuilder, List<String> columns, Object inputValue, Object outputValue){
		List<String> values = Arrays.asList(ValueUtil.formatValue(inputValue), ValueUtil.formatValue(outputValue));

		return DOMUtil.createRow(documentBuilder, columns, values);
	}

	public Map<?, ?> getMapping(){
		return get("mapping", Map.class);
	}

	public Object getDefaultValue(){
		return get("default_value");
	}
}