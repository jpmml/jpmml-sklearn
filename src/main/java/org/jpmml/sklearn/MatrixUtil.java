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

public class MatrixUtil {

	private MatrixUtil(){
	}

	/**
	 * @param values A row-major matrix.
	 */
	static
	public <E> List<E> getRow(List<E> values, int rows, int columns, int row){
		ClassDictUtil.checkSize((rows * columns), values);

		int offset = (row * columns);

		return values.subList(offset, offset + columns);
	}

	/**
	 * @param values A row-major matrix.
	 */
	static
	public <E> List<E> getColumn(List<E> values, int rows, int columns, int column){
		ClassDictUtil.checkSize((rows * columns), values);

		List<E> result = new ArrayList<>(rows);

		for(int row = 0; row < rows; row++){
			E value = values.get((row * columns) + column);

			result.add(value);
		}

		return result;
	}
}