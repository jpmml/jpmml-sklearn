/*
 * Copyright (c) 2019 Villu Ruusmann
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

import java.util.HashMap;

import numpy.DType;
import org.jpmml.sklearn.CClassDict;

public class MaskedArray extends CClassDict {

	public MaskedArray(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){

		if(args.length == 4){
			HashMap<String, Object> values = createAttributeMap(
				new String[]{"data", "mask"},
				new Object[]{new NDArray(), new NDArray()}
			);

			super.__setstate__(values);

			return;
		}

		super.__init__(args);
	}

	@Override
	public void __setstate__(Object[] args){

		if(args.length == 7){
			NDArray data = getData();
			data.__setstate__(new Object[]{null, args[1], args[2], args[3], args[4]});

			NDArray mask = getMask();
			mask.__setstate__(new Object[]{null, args[1], make_mask_descr((DType)args[2]), args[3], args[5]});

			setFillValue(args[6]);

			return;
		}

		super.__setstate__(args);
	}

	public NDArray getData(){
		return (NDArray)get("data");
	}

	public NDArray getMask(){
		return (NDArray)get("mask");
	}

	public Object getFillValue(){
		return get("fill_value");
	}

	public MaskedArray setFillValue(Object fillValue){
		put("fill_value", fillValue);

		return this;
	}

	static
	private DType make_mask_descr(DType ndtype){
		// XXX
		return ndtype;
	}
}