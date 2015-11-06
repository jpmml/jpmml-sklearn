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
package org.jpmml.sklearn;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.FieldName;

public class Schema {

	private FieldName targetField = null;

	private List<String> targetCategories = null;

	private List<FieldName> activeFields = null;


	public Schema(List<FieldName> activeFields){
		this(null, null, activeFields);
	}

	public Schema(FieldName targetField, List<FieldName> activeFields){
		this(targetField, null, activeFields);
	}

	public Schema(FieldName targetField, List<String> targetCategories, List<FieldName> activeFields){
		setTargetField(targetField);
		setTargetCategories(targetCategories);
		setActiveFields(activeFields);
	}

	public FieldName getTargetField(){
		return this.targetField;
	}

	private void setTargetField(FieldName targetField){
		this.targetField = targetField;
	}

	public List<String> getTargetCategories(){
		return this.targetCategories;
	}

	private void setTargetCategories(List<String> targetCategories){
		this.targetCategories = targetCategories;
	}

	public FieldName getActiveField(int index){
		List<FieldName> activeFields = getActiveFields();

		return activeFields.get(index);
	}

	public List<FieldName> getActiveFields(){
		return this.activeFields;
	}

	private void setActiveFields(List<FieldName> activeFields){
		this.activeFields = activeFields;
	}

	static
	public FieldName createTargetField(){
		return FieldName.create("y");
	}

	static
	public List<FieldName> createActiveFields(int size){
		List<FieldName> result = new ArrayList<>();

		for(int i = 0; i < size; i++){
			result.add(FieldName.create("x" + String.valueOf(i + 1)));
		}

		return result;
	}
}