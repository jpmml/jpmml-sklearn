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
import java.util.List;

import com.google.common.base.Charsets;
import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.sklearn.CClassDict;
import org.jpmml.sklearn.HasArray;

public class NDArray extends CClassDict implements HasArray {

	private Object content = null;


	public NDArray(){
		this("numpy", "ndarray");
	}

	public NDArray(String module, String name){
		super(module, name);
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

	@Override
	public List<?> getArrayContent(){
		return NDArrayUtil.getContent(this);
	}

	@Override
	public int[] getArrayShape(){
		return NDArrayUtil.getShape(this);
	}

	public Object getContent(){

		if(this.content == null){
			this.content = loadContent();
		}

		return this.content;
	}

	private Object loadContent(){
		Object[] shape = getShape();
		Object descr = getDescr();
		Object data = getData();

		if(!(data instanceof byte[])){
			return data;
		}

		try(InputStream is = new ByteArrayInputStream((byte[])data)){
			return NDArrayUtil.parseData(is, descr, shape);
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
		Object data = get("data");

		if(data instanceof String){
			String string = (String)data;

			return string.getBytes(Charsets.ISO_8859_1);
		}

		return data;
	}

	private static final String[] SETSTATE_ATTRIBUTES = {
		"version",
		"shape",
		"descr",
		"fortran_order",
		"data"
	};
}
