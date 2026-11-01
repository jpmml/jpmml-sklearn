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

import java.util.List;
import java.util.Objects;

import org.jpmml.evaluator.Table;

public class ExceptionAssignment implements TableMutation {

	private List<Integer> rows = null;

	private Exception exception = null;


	public ExceptionAssignment(List<Integer> rows, Exception exception){
		setRows(rows);
		setException(exception);
	}

	@Override
	public Table apply(Table table){
		List<Integer> rows = getRows();
		Exception exception = getException();

		for(Integer row : rows){
			table.setException(row, exception);
		}

		return table;
	}

	public List<Integer> getRows(){
		return this.rows;
	}

	private void setRows(List<Integer> rows){
		this.rows = Objects.requireNonNull(rows);
	}

	public Exception getException(){
		return this.exception;
	}

	private void setException(Exception exception){
		this.exception = Objects.requireNonNull(exception);
	}

	static
	public boolean matches(Exception expectedException, Exception exception){
		Class<? extends Exception> expectedClazz = expectedException.getClass();

		if(expectedClazz.isInstance(exception)){
			String expectedMessage = expectedException.getMessage();

			if(expectedMessage != null){
				String message = exception.getMessage();

				return Objects.equals(expectedMessage, message);
			}

			return true;
		}

		return false;
	}
}
