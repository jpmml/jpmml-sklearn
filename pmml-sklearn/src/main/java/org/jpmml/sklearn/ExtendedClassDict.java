/*
 * Copyright (c) 2025 Villu Ruusmann
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

import java.util.Map;

import net.razorvine.pickle.IObjectConstructor;
import net.razorvine.pickle.objects.ClassDict;
import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.python.Attribute;
import org.jpmml.python.CastUtil;
import org.jpmml.python.Castable;
import org.jpmml.python.InvalidAttributeException;
import org.jpmml.python.PickleUtil;
import sklearn2pmml.SkLearn2PMMLFields;

public class ExtendedClassDict extends ClassDict implements Castable {

	public ExtendedClassDict(String module, String name){
		super(module, name);
	}

	@Override
	public Object castTo(Class<?> clazz){
		ClassDictConstructor dictConstructor = (ClassDictConstructor)getObjectConstructor();

		ClassDict dict = (ClassDict)dictConstructor.construct(new Object[0]);
		dict.__setstate__(this);

		Object object = CastUtil.deepCastTo(dict, clazz);

		return clazz.cast(object);
	}

	private IObjectConstructor getObjectConstructor(){
		Object pmmlBaseClass = get(SkLearn2PMMLFields.PMML_BASE_CLASS);

		if(pmmlBaseClass instanceof ClassDictConstructor){
			ClassDictConstructor dictConstructor = (ClassDictConstructor)pmmlBaseClass;

			return dictConstructor;
		}

		Map<String, IObjectConstructor> objectConstructors = PickleUtil.getObjectConstructors();

		IObjectConstructor objectConstructor = objectConstructors.get(pmmlBaseClass);
		if(objectConstructor == null){
			Attribute attribute = new Attribute(this, SkLearn2PMMLFields.PMML_BASE_CLASS);

			throw new InvalidAttributeException("Attribute \'" + attribute.format() + "\' refers to an unknown Python class " + pmmlBaseClass, attribute);
		}

		return objectConstructor;
	}

	static
	public ExtendedClassDict build(String className){
		String module;
		String name;

		int dot = className.lastIndexOf('.');
		if(dot > -1){
			module = className.substring(0, dot);
			name = className.substring(dot + 1);
		} else

		{
			module = null;
			name = className;
		}

		return new ExtendedClassDict(module, name);
	}
}