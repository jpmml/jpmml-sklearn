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

import org.jpmml.python.ClassDictUtil;
import sklearn2pmml.preprocessing.ExpressionTransformer;

public class TransformerCastFunction<E extends Transformer> extends StepCastFunction<E> {

	public TransformerCastFunction(Class<? extends E> clazz){
		super(clazz);
	}

	@Override
	protected String formatMessage(Object object){
		return "The object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer";
	}

	@Override
	protected String formatSolution(Object object){
		String className = getClassName(object);

		if(className != null){
			return "Replace this transformer with the " + (ExpressionTransformer.class).getName() + " transformer. Alternatively, develop and register a custom JPMML-SkLearn converter";
		}

		return super.formatSolution(object);
	}

	@Override
	protected String formatExample(Object object){
		String className = getClassName(object);

		if(className != null){
			return ExpressionTransformer.formatSimpleExample();
		}

		return super.formatExample(object);
	}
}