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
package numpy.core;

import java.util.Arrays;

import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.sklearn.CClassDict;

public class NDArray extends CClassDict {

	public NDArray(){
		super("numpy", "ndarray");
	}

	@Override
	public void __init__(Object[] args){

		if(args.length == 3){

			// https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/externals/joblib/numpy_pickle.py
			if((args[0] instanceof ClassDictConstructor) && (args[1] instanceof Object[] && Arrays.equals((Object[])args[1], new Object[]{0})) && (args[2] instanceof byte[] && Arrays.equals((byte[])args[2], new byte[]{(byte)'b'}))){
				return;
			}
		}

		super.__init__(args);
	}

	/**
	 * https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/methods.c
	 */
	@Override
	public void __setstate__(Object[] args){
		super.__setstate__(createAttributeMap(SETSTATE_ATTRIBUTES, args));
	}

	public Object getData(){
		return get("data");
	}

	private static final String[] SETSTATE_ATTRIBUTES = {
		"version",
		"shape",
		"descr",
		"fortran_order",
		"data"
	};
}