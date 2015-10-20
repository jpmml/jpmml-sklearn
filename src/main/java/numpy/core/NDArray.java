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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.sklearn.CClassDict;

public class NDArray extends CClassDict {

	private Object content = null;


	public NDArray(){
		super("numpy", "ndarray");
	}

	@Override
	public void __init__(Object[] args){

		if(isDefault(args)){

			// XXX
			return;
		}

		super.__init__(args);
	}

	private boolean isDefault(Object[] args){

		if(args.length != 3){
			return false;
		} // End if

		if(args[0] instanceof ClassDictConstructor){

			if((args[1] instanceof Object[] && Arrays.equals((Object[])args[1], new Object[]{0}))){

				// Python 2(.7)
				if((args[2] instanceof String) && ((String)args[2]).equals("b")){
					return true;
				} // End if

				// Python 3(.4)
				if((args[2] instanceof byte[] && Arrays.equals((byte[])args[2], new byte[]{(byte)'b'}))){
					return true;
				}
			}
		}

		return false;
	}

	/**
	 * https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/methods.c
	 */
	@Override
	public void __setstate__(Object[] args){
		super.__setstate__(createAttributeMap(SETSTATE_ATTRIBUTES, args));
	}

	public Object getContent(){

		if(this.content == null){
			Object data = getData();

			this.content = ((data instanceof byte[]) ? loadContent() : data);
		}

		return this.content;
	}

	private Object loadContent(){
		Object[] shape = getShape();
		Object descr = getDescr();
		byte[] data = (byte[])getData();

		try {
			InputStream is = new ByteArrayInputStream(data);

			try {
				return NDArrayUtil.parseData(is, descr, shape);
			} finally {
				is.close();
			}
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}
	}

	public Object[] getShape(){
		return (Object[])get("shape");
	}

	public Object getDescr(){
		return get("descr");
	}

	public Boolean getFortranOrder(){
		return (Boolean)get("fortran_order");
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