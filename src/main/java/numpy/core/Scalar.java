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
import java.util.List;

import com.google.common.base.Charsets;
import com.google.common.collect.Iterables;
import numpy.DType;
import org.jpmml.sklearn.CClassDict;
import org.jpmml.sklearn.HasArray;

public class Scalar extends CClassDict implements HasArray {

	private List<?> content = null;


	public Scalar(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){
		super.__setstate__(createAttributeMap(INIT_ATTRIBUTES, args));
	}

	@Override
	public List<?> getArrayContent(){
		List<?> content = getContent();

		return content;
	}

	@Override
	public int[] getArrayShape(){
		List<?> content = getContent();

		return new int[]{content.size()};
	}

	public Object getOnlyElement(){
		List<?> content = getContent();

		return Iterables.getOnlyElement(content);
	}

	public List<?> getContent(){

		if(this.content == null){
			this.content = loadContent();
		}

		return this.content;
	}

	private List<?> loadContent(){
		DType dtype = getDType();
		byte[] obj = getObj();

		try(InputStream is = new ByteArrayInputStream(obj)){
			return (List<?>)NDArrayUtil.parseData(is, dtype, new Object[0]);
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}
	}

	public DType getDType(){
		return (DType)get("dtype");
	}

	public byte[] getObj(){
		Object obj = get("obj");

		if(obj instanceof String){
			String string = (String)obj;

			return string.getBytes(Charsets.ISO_8859_1);
		}

		return (byte[])obj;
	}

	private static final String[] INIT_ATTRIBUTES = {
		"dtype",
		"obj"
	};
}
