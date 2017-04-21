/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import numpy.core.UFunc;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class FunctionTransformer extends Transformer {

	public FunctionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Object func = getFunc();

		UFunc ufunc;

		try {
			ufunc = (UFunc)func;
		} catch(ClassCastException cce){
			throw new IllegalArgumentException("The function object (" + ClassDictUtil.formatClass(func) + ") is not a Numpy universal function", cce);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			ContinuousFeature continuousFeature = (features.get(i)).toContinuousFeature();

			FieldName name = FeatureUtil.createName(ufunc.getName(), continuousFeature);

			DerivedField derivedField = encoder.getDerivedField(name);
			if(derivedField == null){
				Expression expression = encodeUFunc(ufunc, continuousFeature.ref());

				derivedField = encoder.createDerivedField(name, expression);
			}

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public Object getFunc(){
		return get("func");
	}

	static
	Expression encodeUFunc(UFunc ufunc, FieldRef fieldRef){
		String name = ufunc.getName();

		switch(name){
			case "absolute":
				return PMMLUtil.createApply("abs", fieldRef);
			case "ceil":
			case "exp":
			case "floor":
				return PMMLUtil.createApply(name, fieldRef);
			case "log":
				return PMMLUtil.createApply("ln", fieldRef);
			case "log10":
				return PMMLUtil.createApply(name, fieldRef);
			case "negative":
				return PMMLUtil.createApply("*", PMMLUtil.createConstant(-1), fieldRef);
			case "reciprocal":
				return PMMLUtil.createApply("/", PMMLUtil.createConstant(1), fieldRef);
			case "sign":
				return PMMLUtil.createApply("if", PMMLUtil.createApply("lessThan", fieldRef, PMMLUtil.createConstant(0)),
					PMMLUtil.createConstant(-1), // x < 0
					PMMLUtil.createApply("if", PMMLUtil.createApply("greaterThan", fieldRef, PMMLUtil.createConstant(0)),
						PMMLUtil.createConstant(+1), // x > 0
						PMMLUtil.createConstant(0) // x == 0
					)
				);
			case "sqrt":
				return PMMLUtil.createApply(name, fieldRef);
			case "square":
				return PMMLUtil.createApply("*", fieldRef, fieldRef);
			default:
				break;
		}

		switch(name){
			case "asin":
			case "acos":
			case "atan":
			case "atan2":
			case "cos":
			case "cosh":
			case "expm1":
			case "hypot":
				return PMMLUtil.createApply("x-" + name, fieldRef);
			case "log1p":
				return PMMLUtil.createApply("x-ln1p", fieldRef);
			case "rint":
			case "sin":
			case "sinh":
			case "tan":
			case "tanh":
				return PMMLUtil.createApply("x-" + name, fieldRef);
			default:
				throw new IllegalArgumentException(name);
		}
	}
}