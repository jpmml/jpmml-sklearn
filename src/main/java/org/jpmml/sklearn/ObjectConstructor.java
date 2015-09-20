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

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

import net.razorvine.pickle.IObjectConstructor;
import net.razorvine.pickle.PickleException;
import net.razorvine.pickle.objects.ClassDict;

public class ObjectConstructor implements IObjectConstructor {

	private String module = null;

	private String name = null;

	private Class<? extends ClassDict> clazz = null;


	public ObjectConstructor(String module, String name){
		this(module, name, null);
	}

	public ObjectConstructor(String module, String name, Class<? extends ClassDict> clazz){
		setModule(module);
		setName(name);
		setClazz(clazz);
	}

	public ClassDict newObject(){
		Class<? extends ClassDict> clazz = getClazz();

		if(clazz == null){
			throw new RuntimeException();
		}

		try {
			try {
				Constructor<? extends ClassDict> namedConstructor = clazz.getConstructor(String.class, String.class);

				return namedConstructor.newInstance(getModule(), getName());
			} catch(NoSuchMethodException nsme){
				return clazz.newInstance();
			}
		} catch(IllegalAccessException | InstantiationException | InvocationTargetException e){
			throw new RuntimeException(e);
		}
	}

	@Override
	public ClassDict construct(Object[] args){

		if(args.length != 0){
			throw new PickleException(Arrays.deepToString(args));
		}

		return newObject();
	}

	public String getModule(){
		return this.module;
	}

	private void setModule(String module){
		this.module = module;
	}

	public String getName(){
		return this.name;
	}

	private void setName(String name){
		this.name = name;
	}

	public Class<? extends ClassDict> getClazz(){
		return this.clazz;
	}

	private void setClazz(Class<? extends ClassDict> clazz){
		this.clazz = clazz;
	}
}