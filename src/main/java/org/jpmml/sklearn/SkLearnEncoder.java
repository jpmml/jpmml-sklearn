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

import java.lang.reflect.Field;
import java.util.LinkedHashSet;
import java.util.Set;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelEncoder;

public class SkLearnEncoder extends ModelEncoder {

	private Set<FieldName> frozenFields = new LinkedHashSet<>();


	public DataField createDataField(FieldName name){
		return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
	}

	public DerivedField createDerivedField(FieldName name, Expression expression){
		return createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
	}

	public void renameFeature(Feature feature, FieldName renamedName){
		FieldName name = feature.getName();

		org.dmg.pmml.Field<?> pmmlField = getField(name);

		if(pmmlField instanceof DataField){
			throw new IllegalArgumentException("User input field " + name.getValue() + " cannot be renamed");
		}

		DerivedField derivedField = removeDerivedField(name);

		try {
			Field field = Feature.class.getDeclaredField("name");

			if(!field.isAccessible()){
				field.setAccessible(true);
			}

			field.set(feature, renamedName);
		} catch(ReflectiveOperationException roe){
			throw new RuntimeException(roe);
		}

		derivedField.setName(renamedName);

		addDerivedField(derivedField);
	}

	public boolean isFrozen(FieldName name){
		return this.frozenFields.contains(name);
	}

	public void setFrozen(FieldName name, boolean frozen){

		if(frozen){
			this.frozenFields.add(name);
		} else

		{
			this.frozenFields.remove(name);
		}
	}
}