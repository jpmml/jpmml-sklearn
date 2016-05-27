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

import java.util.Arrays;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Row;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.DOMUtil;
import sklearn.OneToOneTransformer;

public class LabelEncoder extends OneToOneTransformer {

	public LabelEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		return DataType.STRING;
	}

	@Override
	public Expression encode(FieldName name){
		MapValues mapValues = new MapValues()
			.addFieldColumnPairs(new FieldColumnPair(name, "input"))
			.setOutputColumn("output");

		DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

		InlineTable inlineTable = new InlineTable();

		List<String> keys = Arrays.asList("input", "output");

		List<?> classes = getClasses();

		for(int i = 0; i < classes.size(); i++){
			List<String> values = Arrays.asList(ValueUtil.formatValue(classes.get(i)), String.valueOf(i));

			Row row = DOMUtil.createRow(documentBuilder, keys, values);

			inlineTable.addRows(row);
		}

		mapValues.setInlineTable(inlineTable);

		return mapValues;
	}

	@Override
	public List<?> getClasses(){
		return ClassDictUtil.getArray(this, "classes_");
	}
}