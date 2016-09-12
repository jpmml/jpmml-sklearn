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
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import sklearn.Transformer;

public class FunctionTransformer extends Transformer {

	public FunctionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> inputFeatures, FeatureMapper featureMapper){
		Object func = getFunc();

		if(ids.size() != inputFeatures.size()){
			throw new IllegalArgumentException();
		}

		UFunc ufunc;

		try {
			ufunc = (UFunc)func;
		} catch(ClassCastException cce){
			throw new IllegalArgumentException("The function object (" + ClassDictUtil.formatClass(func) + ") is not a Numpy universal function", cce);
		}

		List<Feature> features = new ArrayList<>();

		for(int i = 0; i < inputFeatures.size(); i++){
			String id = ids.get(i);
			Feature inputFeature = inputFeatures.get(i);

			Expression expression = encodeUFunc(ufunc, inputFeature.ref());

			DerivedField derivedField = featureMapper.createDerivedField(FieldName.create(ufunc.getName() + "(" + id + ")"), expression);

			features.add(new ContinuousFeature(derivedField));
		}

		return features;
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
			case "rint":
				return PMMLUtil.createApply("round", fieldRef);
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
				throw new IllegalArgumentException(name);
		}
	}
}