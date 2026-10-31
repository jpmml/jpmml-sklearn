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
package sklearn;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import net.razorvine.pickle.IObjectConstructor;
import net.razorvine.pickle.objects.ClassDict;
import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.python.CastFunction;
import org.jpmml.python.CastUtil;
import org.jpmml.python.ClassDictConstructorUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.PythonObjectConstructor;
import org.jpmml.sklearn.SkLearnException;

public class StepCastFunction<E extends Step> extends CastFunction<E> {

	public StepCastFunction(Class<? extends E> clazz){
		super(clazz);
	}

	@Override
	public E apply(Object object){
		Class<? extends E> clazz = getClazz();

		object = CastUtil.deepCastTo(object, clazz);

		try {
			return super.apply(object);
		} catch(ClassCastException cce){

			if(object instanceof Transformer){
				Transformer transformer = (Transformer)object;

				throw new TransformerCastException(transformer, Collections.singletonList(clazz), cce);
			} else

			if(object instanceof Estimator){
				Estimator estimator = (Estimator)object;

				throw new EstimatorCastException(estimator, Collections.singletonList(clazz), cce);
			} else

			{
				throw new SkLearnException(formatMessage(object), cce)
					.setSolution(formatSolution(object))
					.setExample(formatExample(object));
			}
		}
	}

	protected String formatMessage(Object object){
		return "The object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer or Estimator";
	}

	protected String formatSolution(Object object){
		String className = getClassName(object);

		if(className != null){
			List<PythonObjectConstructor> subclassConstructors = getInstantiableSubclassConstructors(className);

			if(!subclassConstructors.isEmpty()){
				return "Replace this class with one of its supported subclasses " + formatClassNames(subclassConstructors);
			}

			return "Develop and register a custom JPMML-SkLearn converter";
		}

		return null;
	}

	protected String formatExample(Object object){
		return null;
	}

	static
	protected List<PythonObjectConstructor> getInstantiableSubclassConstructors(String className){
		Class<?> clazz;

		try {
			clazz = Class.forName(className);
		} catch(ClassNotFoundException cnfe){
			return Collections.emptyList();
		}

		if(!Modifier.isAbstract(clazz.getModifiers())){
			return Collections.emptyList();
		}

		List<PythonObjectConstructor> result = new ArrayList<>();

		Map<String, IObjectConstructor> objectConstructors = PickleUtil.getObjectConstructors();

		Collection<IObjectConstructor> values = objectConstructors.values();
		for(IObjectConstructor value : values){

			if(value instanceof PythonObjectConstructor){
				PythonObjectConstructor pythonObjectConstructor = (PythonObjectConstructor)value;

				Class<?> subclazz = pythonObjectConstructor.getClazz();

				if((clazz != subclazz) && clazz.isAssignableFrom(subclazz)){

					if(!Modifier.isAbstract(subclazz.getModifiers())){
						result.add(pythonObjectConstructor);
					}
				}
			}
		}

		return result;
	}

	static
	protected String formatClassNames(List<? extends ClassDictConstructor> dictConstructors){
		List<String> classNames = dictConstructors.stream()
			.map(ClassDictConstructorUtil::getClassName)
			.sorted()
			.collect(Collectors.toList());

		return ExceptionUtil.formatNames(classNames);
	}

	static
	protected String getClassName(Object object){

		if(object instanceof ClassDict){
			ClassDict dict = (ClassDict)object;

			return dict.getClassName();
		} else

		if(object instanceof ClassDictConstructor){
			ClassDictConstructor dictConstructor = (ClassDictConstructor)object;

			return ClassDictConstructorUtil.getClassName(dictConstructor);
		}

		return null;
	}
}