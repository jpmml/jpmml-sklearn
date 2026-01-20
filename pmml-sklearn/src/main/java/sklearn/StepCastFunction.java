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

import net.razorvine.pickle.objects.ClassDict;
import net.razorvine.pickle.objects.ClassDictConstructor;
import org.jpmml.python.CastFunction;
import org.jpmml.python.CastUtil;
import org.jpmml.python.ClassDictConstructorUtil;
import org.jpmml.python.ClassDictUtil;
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
			throw new SkLearnException(formatMessage(object))
				.setSolution(formatSolution(object))
				.setExample(formatExample(object));
		}
	}

	protected String formatMessage(Object object){
		return "The object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer or Estimator";
	}

	protected String formatSolution(Object object){
		String className = getClassName(object);

		if(className != null){
			return "Develop and register a custom JPMML-SkLearn converter";
		}

		return null;
	}

	protected String formatExample(Object object){
		return null;
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