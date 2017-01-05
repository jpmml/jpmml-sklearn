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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Row;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn.TypeUtil;

public class LabelEncoder extends Transformer {

	public LabelEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		List<?> classes = getClasses();

		return TypeUtil.getDataType(classes, DataType.STRING);
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> inputFeatures, SkLearnEncoder encoder){
		List<?> classes = getClasses();

		if(ids.size() != 1 || inputFeatures.size() != 1){
			throw new IllegalArgumentException();
		}

		String id = ids.get(0);
		Feature inputFeature = inputFeatures.get(0);

		DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

		InlineTable inlineTable = new InlineTable();

		List<String> columns = Arrays.asList("input", "output");

		List<String> categories = new ArrayList<>();

		for(int i = 0; i < classes.size(); i++){
			String category = ValueUtil.formatValue(classes.get(i));

			categories.add(category);

			List<String> values = Arrays.asList(category, String.valueOf(i));

			Row row = DOMUtil.createRow(documentBuilder, columns, values);

			inlineTable.addRows(row);
		}

		encoder.updateValueSpace(inputFeature.getName(), categories);

		MapValues mapValues = new MapValues()
			.addFieldColumnPairs(new FieldColumnPair(inputFeature.getName(), columns.get(0)))
			.setOutputColumn(columns.get(1))
			.setInlineTable(inlineTable);

		DerivedField derivedField = encoder.createDerivedField(createName(id), OpType.CATEGORICAL, DataType.INTEGER, mapValues);

		return Collections.<Feature>singletonList(new CategoricalFeature(encoder, derivedField, categories));
	}

	public List<?> getClasses(){
		return ClassDictUtil.getArray(this, "classes_");
	}
}