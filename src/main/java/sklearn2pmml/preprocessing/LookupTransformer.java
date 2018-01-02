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

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		List<Object> outputValues = new ArrayList<>(mapping.size() + 1);

		DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

		InlineTable inlineTable = new InlineTable();

		List<String> columns = Arrays.asList("input", "output");

		Collection<? extends Map.Entry<?, ?>> entries = mapping.entrySet();
		for(Map.Entry<?, ?> entry : entries){
			Object inputValue = entry.getKey();
			Object outputValue = entry.getValue();

			if(outputValue == null){
				continue;
			}

			List<String> values = Arrays.asList(ValueUtil.formatValue(inputValue), ValueUtil.formatValue(outputValue));

			Row row = DOMUtil.createRow(documentBuilder, columns, values);

			inlineTable.addRows(row);

			outputValues.add(outputValue);
		}

		MapValues mapValues = new MapValues()
			.addFieldColumnPairs(new FieldColumnPair(feature.getName(), columns.get(0)))
			.setOutputColumn(columns.get(1))
			.setInlineTable(inlineTable);

		if(defaultValue != null){
			mapValues.setDefaultValue(ValueUtil.formatValue(defaultValue));

			outputValues.add(defaultValue);
		}

		DerivedField derivedField = encoder.createDerivedField(FeatureUtil.createName("lookup", feature), OpType.CATEGORICAL, TypeUtil.getDataType(outputValues, DataType.STRING), mapValues);

		feature = new Feature(encoder, derivedField.getName(), derivedField.getDataType()){

			@Override
			public ContinuousFeature toContinuousFeature(){
				PMMLEncoder encoder = getEncoder();

				DerivedField derivedField = (DerivedField)encoder.toContinuous(getName());

				return new ContinuousFeature(encoder, derivedField);
			}
		};

		return Collections.singletonList(feature);
	}

	public Map<?, ?> getMapping(){
		return get("mapping", Map.class);
	}

	public Object getDefaultValue(){
		return get("default_value");
	}
}