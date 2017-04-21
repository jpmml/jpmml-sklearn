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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import joblib.NDArrayWrapper;
import net.razorvine.pickle.objects.ClassDict;
import numpy.core.NDArray;
import numpy.core.NDArrayUtil;

public class ClassDictUtil {

	private ClassDictUtil(){
	}

	static
	public List<?> getArray(ClassDict dict, String name){
		Object object = dict.get(name);

		if(object instanceof HasArray){
			HasArray hasArray = (HasArray)object;

			return hasArray.getArrayContent();
		} // End if

		if(object instanceof Number){
			return Collections.singletonList(object);
		}

		throw new IllegalArgumentException("The value of the " + ClassDictUtil.formatMember(dict, name) + " attribute (" + ClassDictUtil.formatClass(object) + ") is not a supported array type");
	}

	static
	public List<?> getArray(ClassDict dict, String name, String key){
		Object object = dict.get(name);

		if(object instanceof NDArrayWrapper){
			NDArrayWrapper arrayWrapper = (NDArrayWrapper)object;

			object = arrayWrapper.getContent();
		} // End if

		if(object instanceof NDArray){
			NDArray array = (NDArray)object;

			return NDArrayUtil.getContent(array, key);
		}

		throw new IllegalArgumentException("The value of the " + ClassDictUtil.formatMember(dict, name) + " attribute (" + ClassDictUtil.formatClass(object) + ") is not a supported array type");
	}

	static
	public int[] getShape(ClassDict dict, String name, int length){
		int[] shape = getShape(dict, name);

		if(shape.length != length){
			throw new IllegalArgumentException("The dimensionality of the " + ClassDictUtil.formatMember(dict, name) + " attribute (" + shape.length + ") is not " + length);
		}

		return shape;
	}

	static
	public int[] getShape(ClassDict dict, String name){
		Object object = dict.get(name);

		if(object instanceof HasArray){
			HasArray hasArray = (HasArray)object;

			return hasArray.getArrayShape();
		} // End if

		if(object instanceof Number){
			return new int[]{1};
		}

		throw new IllegalArgumentException("The value of the " + ClassDictUtil.formatMember(dict, name) + " attribute (" + ClassDictUtil.formatClass(object) +") is not a supported array type");
	}

	static
	public void checkSize(Collection<?>... collections){
		Collection<?> prevCollection = null;

		for(Collection<?> collection : collections){

			if(collection == null){
				continue;
			} // End if

			if(prevCollection != null && collection.size() != prevCollection.size()){
				throw new IllegalArgumentException("Expected the same number of elements, got different numbers of elements");
			}

			prevCollection = collection;
		}
	}

	static
	public void checkSize(int size, Collection<?>... collections){

		for(Collection<?> collection : collections){

			if(collection == null){
				continue;
			} // End if

			if(collection.size() != size){
				throw new IllegalArgumentException("Expected " + size + " element(s), got " + collection.size() + " element(s)");
			}
		}
	}

	static
	public String formatMember(ClassDict dict, String name){
		String clazz = (String)dict.get("__class__");

		return (clazz + "." + name);
	}

	static
	public String formatClass(Object object){

		if(object == null){
			return null;
		} // End if

		if(object instanceof ClassDict){
			ClassDict dict = (ClassDict)object;

			String clazz = (String)dict.get("__class__");

			return "Python class " + clazz;
		}

		Class<?> clazz = object.getClass();

		return "Java class " + clazz.getName();
	}

	static
	public String toString(ClassDict dict){
		StringBuffer sb = new StringBuffer();

		sb.append("\n{\n");

		String sep = "";

		List<? extends Map.Entry<String, ?>> entries = new ArrayList<>(dict.entrySet());

		Comparator<Map.Entry<String, ?>> comparator = new Comparator<Map.Entry<String, ?>>(){

			@Override
			public int compare(Map.Entry<String, ?> left, Map.Entry<String, ?> right){
				return (left.getKey()).compareToIgnoreCase(right.getKey());
			}
		};
		Collections.sort(entries, comparator);

		for(Map.Entry<String, ?> entry : entries){
			sb.append(sep);

			sep = "\n";

			String key = entry.getKey();
			Object value = entry.getValue();

			sb.append("\t" + key + "=" + value + (" // " + (value != null ? (value.getClass()).getName() : "N/A")));
		}

		sb.append("\n}\n");

		return sb.toString();
	}
}
