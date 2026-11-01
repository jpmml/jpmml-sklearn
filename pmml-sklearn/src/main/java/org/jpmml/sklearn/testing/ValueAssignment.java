/*
 * Copyright (c) 2026 Villu Ruusmann
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
package org.jpmml.sklearn.testing;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import org.jpmml.evaluator.Table;

public class ValueAssignment implements TableMutation {

	private String column = null;

	private List<Integer> rows = null;

	private Object value = null;


	public ValueAssignment(String column, List<Integer> rows, Object value){
		setColumn(column);
		setRows(rows);
		setValue(value);
	}

	@Override
	public Table apply(Table table){
		String column = getColumn();
		List<Integer> rows = getRows();
		Object value = getValue();

		List<?> values = table.getValues(column);

		List<Object> mutatedValues = new ArrayList<>(values);

		for(Integer row : rows){
			mutatedValues.set(row, value);
		}

		table.setValues(column, mutatedValues);

		return table;
	}

	public String getColumn(){
		return this.column;
	}

	private void setColumn(String column){
		this.column = Objects.requireNonNull(column);
	}

	public List<Integer> getRows(){
		return this.rows;
	}

	private void setRows(List<Integer> rows){
		this.rows = Objects.requireNonNull(rows);
	}

	public Object getValue(){
		return this.value;
	}

	private void setValue(Object value){
		this.value = value;
	}
}
