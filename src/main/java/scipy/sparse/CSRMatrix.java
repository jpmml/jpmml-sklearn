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

import java.util.List;

import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.HasArray;
import org.jpmml.sklearn.PyClassDict;

public class CSRMatrix extends PyClassDict implements HasArray {

	public CSRMatrix(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getArrayContent(){
		return CSRMatrixUtil.getContent(this);
	}

	@Override
	public int[] getArrayShape(){
		return CSRMatrixUtil.getShape(this);
	}

	public List<?> getData(){
		return getArray("data");
	}

	public List<Integer> getIndices(){
		return ValueUtil.asIntegers(getArray("indices", Number.class));
	}

	public List<Integer> getIndPtr(){
		return ValueUtil.asIntegers(getArray("indptr", Number.class));
	}

	public Object[] getShape(){
		return (Object[])get("_shape");
	}
}