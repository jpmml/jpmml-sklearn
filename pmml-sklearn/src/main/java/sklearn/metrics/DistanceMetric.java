/*
 * Copyright (c) 2022 Villu Ruusmann
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
package sklearn.metrics;

import java.lang.reflect.Field;

import net.razorvine.pickle.objects.ClassDict;
import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.python.ClassDictConstructorUtil;
import org.jpmml.python.CustomPythonObject;

public class DistanceMetric extends CustomPythonObject {

	public DistanceMetric(String module, String name){
		super(module, name);
	}

	@Override
	public void __init__(Object[] args){

		if(args.length == 1){
			ClassDictConstructor dictConstructor = (ClassDictConstructor)args[0];

			setClassName(ClassDictConstructorUtil.getClassName(dictConstructor));

			return;
		}

		super.__init__(args);
	}

	@Override
	public void __setstate__(Object[] args){
		super.__setstate__(createAttributeMap(SETSTATE_ATTRIBUTES, args));
	}

	private void setClassName(String name){

		try {
			Field classNameField = ClassDict.class.getDeclaredField("classname");
			if(!classNameField.isAccessible()){
				classNameField.setAccessible(true);
			}

			classNameField.set(this, name);
		} catch(ReflectiveOperationException roe){
			throw new RuntimeException(roe);
		}

		put("__class__", name);
	}

	private static final String[] SETSTATE_ATTRIBUTES = {
		"p",
		"vec",
		"mat"
	};
}