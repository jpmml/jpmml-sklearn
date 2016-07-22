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
package joblib;

import java.io.IOException;
import java.io.InputStream;

import net.razorvine.pickle.objects.ClassDict;
import numpy.DType;
import numpy.core.NDArray;
import numpy.core.NDArrayUtil;

public class NumpyArrayWrapper extends ClassDict {

	public NumpyArrayWrapper(String module, String name){
		super(module, name);
	}

	public NDArray toArray(InputStream is) throws IOException {
		DType dtype = getDType();
		Object[] shape = getShape();

		Object descr = dtype.toDescr();
		Boolean fortran_order = parseOrder(getOrder());

		Object data = NDArrayUtil.parseData(is, descr, shape);

		NDArray array = new NDArray();
		array.__setstate__(new Object[]{null, shape, descr, fortran_order, data});

		return array;
	}

	public DType getDType(){
		return (DType)get("dtype");
	}

	public Object[] getShape(){
		return (Object[])get("shape");
	}

	public String getOrder(){
		return (String)get("order");
	}

	static
	private Boolean parseOrder(String order){

		switch(order){
			case "C":
				return Boolean.FALSE;
			case "F":
				return Boolean.TRUE;
			default:
				throw new IllegalArgumentException(order);
		}
	}
}