/*
 * Copyright (c) 2016 Villu Ruusmann
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
package org.jpmml.sklearn;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.TypeDefinitionField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.PMMLUtil;

public class SkLearnEncoder extends ModelEncoder {

	private List<String> ids = new ArrayList<>();

	private List<Feature> features = new ArrayList<>();


	public void updateType(FieldName name, OpType opType, DataType dataType){
		TypeDefinitionField field = getField(name);

		if(opType != null){
			field.setOpType(opType);
		} // End if

		if(dataType != null){
			field.setDataType(dataType);
		}
	}

	public void updateValueSpace(FieldName name, List<String> categories){
		DataField dataField = getDataField(name);

		if(dataField == null){
			throw new IllegalArgumentException(name.getValue());
		}

		List<String> existingCategories = PMMLUtil.getValues(dataField);
		if(existingCategories != null && existingCategories.size() > 0){

			if((existingCategories).equals(categories)){
				return;
			}

			throw new IllegalArgumentException("Data field " + name.getValue() + " has valid values " + existingCategories);
		}

		PMMLUtil.addValues(dataField, categories);
	}

	public DataField createDataField(FieldName name){
		return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
	}

	public DerivedField createDerivedField(FieldName name, Expression expression){
		return createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
	}

	public void addRow(List<String> ids, List<Feature> features){
		ClassDictUtil.checkSize(ids, features);

		this.ids.addAll(ids);
		this.features.addAll(features);
	}

	public List<String> getIds(){
		return this.ids;
	}

	public List<Feature> getFeatures(){
		return this.features;
	}
}