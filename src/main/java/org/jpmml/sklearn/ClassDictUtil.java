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

import net.razorvine.pickle.objects.ClassDict;

public class ClassDictUtil {

	private ClassDictUtil(){
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
	public void checkShapes(int axis, int[]... shapes){
		int[] prevShape = null;

		for(int[] shape : shapes){

			if(prevShape != null && prevShape[axis] != shape[axis]){
				throw new IllegalArgumentException("Expected the same number of elements, got different number of elements");
			}

			prevShape = shape;
		}
	}

	static
	public void checkShapes(int axis, int size, int[]... shapes){

		for(int[] shape : shapes){

			if(shape[axis] != size){
				throw new IllegalArgumentException("Expected " + size + " element(s), got " + shape[axis] + " element(s)");
			}
		}
	}

	static
	public String getName(ClassDict dict){
		String clazz = (String)dict.get("__class__");

		if(clazz == null){
			throw new IllegalArgumentException();
		}

		return clazz;
	}

	static
	public String getSimpleName(ClassDict dict){
		String name = getName(dict);

		int dot = name.lastIndexOf('.');
		if(dot > -1){
			return name.substring(dot + 1);
		}

		return name;
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

			return "Python class " + getName(dict);
		}

		Class<?> clazz = object.getClass();

		return "Java class " + clazz.getName();
	}

	static
	public String formatProxyExample(Class<? extends ClassDict> proxyClazz, ClassDict dict){
		return (proxyClazz.getSimpleName() + "(" + ClassDictUtil.getSimpleName(dict) + "(...))");
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
