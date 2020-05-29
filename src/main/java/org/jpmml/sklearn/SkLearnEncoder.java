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
import java.util.LinkedHashMap;
import java.util.Map;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.PythonEncoder;
import sklearn2pmml.decoration.Alias;
import sklearn2pmml.decoration.Domain;

public class SkLearnEncoder extends PythonEncoder {

	private Map<FieldName, Domain> domains = new LinkedHashMap<>();


	public SkLearnEncoder(){
	}

	public DataField createDataField(FieldName name){
		return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
	}

	public DerivedField createDerivedField(FieldName name, Expression expression){
		return createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
	}

	@Override
	public void addDerivedField(DerivedField derivedField){

		try {
			super.addDerivedField(derivedField);
		} catch(RuntimeException re){
			FieldName name = derivedField.getName();

			String message = "Field " + name.getValue() + " is already defined. " +
				"Please refactor the pipeline so that it would not contain duplicate field declarations, " +
				"or use the " + (Alias.class).getName() + " wrapper class to override the default name with a custom name (eg. " + Alias.formatAliasExample() + ")";

			throw new IllegalArgumentException(message, re);
		}
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
		return this.domains.containsKey(name);
	}

	public Domain getDomain(FieldName name){
		return this.domains.get(name);
	}

	public void setDomain(FieldName name, Domain domain){

		if(domain != null){
			this.domains.put(name, domain);
		} else

		{
			this.domains.remove(name);
		}
	}

	static {
		PickleUtil.init("sklearn2pmml.properties");
	}
}