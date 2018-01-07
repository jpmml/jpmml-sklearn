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
package org.jpmml.sklearn;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

import joblib.NDArrayWrapper;
import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.objects.ClassDict;
import numpy.core.NDArray;
import numpy.core.NDArrayUtil;

public class CClassDict extends ClassDict {

	public CClassDict(String module, String name){
		super(module, name);

		reset();
	}

	public List<?> getArray(String name){
		Object object = get(name);

		if(object instanceof HasArray){
			HasArray hasArray = (HasArray)object;

			return hasArray.getArrayContent();
		}

		throw new IllegalArgumentException("The value of \'" + ClassDictUtil.formatMember(this, name) + "\' attribute (" + ClassDictUtil.formatClass(object) + ") is not a supported array type");
	}

	public List<?> getArray(String name, String key){
		Object object = get(name);

		if(object instanceof NDArrayWrapper){
			NDArrayWrapper arrayWrapper = (NDArrayWrapper)object;

			object = arrayWrapper.getContent();
		} // End if

		if(object instanceof NDArray){
			NDArray array = (NDArray)object;

			return NDArrayUtil.getContent(array, key);
		}

		throw new IllegalArgumentException("The value of \'" + ClassDictUtil.formatMember(this, name) + "\' attribute (" + ClassDictUtil.formatClass(object) + ") is not a supported array type");
	}

	public void __init__(Object[] args){

		if(args.length > 0){
			throw new PickleException(ClassDictUtil.formatMember(this, "__init__(" + Arrays.deepToString(args) + ")"));
		}

		reset();
	}

	public void __setstate__(Object[] args){

		if(args.length > 0){
			throw new PickleException(ClassDictUtil.formatMember(this, "__setstate__(" + Arrays.deepToString(args) + ")"));
		}

		reset();
	}

	public HashMap<String, Object> __getstate__(){
		HashMap<String, Object> result = new LinkedHashMap<>(this);
		result.remove("__class__");

		return result;
	}

	@Override
	public void __setstate__(HashMap<String, Object> values){
		HashMap<String, Object> state = __getstate__();

		// The state is additive
		state.putAll(values);

		super.__setstate__(state);
	}

	protected void reset(){
		HashMap<String, Object> state = new HashMap<>();

		super.__setstate__(state);
	}

	static
	public HashMap<String, Object> createAttributeMap(String[] attributes, Object[] args){

		if(attributes.length != args.length){
			throw new PickleException("Expected " + attributes.length + " attribute(s), got " + args.length + " attribute(s)");
		}

		HashMap<String, Object> result = new LinkedHashMap<>();

		for(int i = 0; i < attributes.length; i++){
			result.put(attributes[i], args[i]);
		}

		return result;
	}
}
