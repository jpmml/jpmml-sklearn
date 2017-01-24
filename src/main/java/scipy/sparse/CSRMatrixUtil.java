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
package scipy.sparse;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.common.primitives.Ints;
import org.jpmml.converter.ValueUtil;

public class CSRMatrixUtil {

	private CSRMatrixUtil(){
	}

	static
	public int[] getShape(CSRMatrix matrix){
		Object[] shape = matrix.getShape();

		if(shape.length == 1){
			return new int[]{ValueUtil.asInt((Number)shape[0])};
		} else

		if(shape.length == 2){
			return new int[]{ValueUtil.asInt((Number)shape[0]), ValueUtil.asInt((Number)shape[1])};
		}

		List<? extends Number> values = (List)Arrays.asList(shape);

		return Ints.toArray(ValueUtil.asIntegers(values));
	}

	static
	public List<? extends Number> getContent(CSRMatrix matrix){
		int[] shape = getShape(matrix);

		int numberOfRows = shape[0];
		int numberOfColumns = shape[1];

		List<Number> result = new ArrayList<>(numberOfRows * numberOfColumns);

		List<?> data = matrix.getData();
		List<Integer> indices = matrix.getIndices();
		List<Integer> indPtr = matrix.getIndPtr();

		for(int row = 0; row < numberOfRows; row++){
			int offset = result.size();

			for(int column = 0; column < numberOfColumns; column++){
				result.add(ZERO);
			}

			int begin = indPtr.get(row);
			int end = indPtr.get(row + 1);

			for(int i = begin; i < end; i++){
				int index = indices.get(i);

				// Relative column indices [0, numberOfColumns - 1]
				if(index < numberOfColumns){
					index = (offset + index);
				}

				result.set(index, (Number)data.get(i));
			}
		}

		return result;
	}

	private static final Integer ZERO = Integer.valueOf(0);
}