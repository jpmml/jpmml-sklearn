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
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class FunctionTransformer extends Transformer {

	public FunctionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		UFunc func = getFunc();

		if(func == null){
			return features;
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			ContinuousFeature continuousFeature = (features.get(i)).toContinuousFeature();

			DerivedField derivedField = encoder.ensureDerivedField(FeatureUtil.createName(func.getName(), continuousFeature), OpType.CONTINUOUS, DataType.DOUBLE, () -> encodeUFunc(func, continuousFeature.ref()));

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public UFunc getFunc(){
		return getOptional("func", UFunc.class);
	}

	public UFunc getInverseFunc(){
		return getOptional("inverse_func", UFunc.class);
	}

	static
	public Expression encodeUFunc(UFunc ufunc, FieldRef fieldRef){
		String name = ufunc.getName();

		switch(name){
			case "absolute":
				return PMMLUtil.createApply(PMMLFunctions.ABS, fieldRef);
			case "acos":
				return PMMLUtil.createApply(PMMLFunctions.ACOS, fieldRef);
			case "asin":
				return PMMLUtil.createApply(PMMLFunctions.ASIN, fieldRef);
			case "atan":
				return PMMLUtil.createApply(PMMLFunctions.ATAN, fieldRef);
			case "atan2":
				return PMMLUtil.createApply(PMMLFunctions.ATAN2, fieldRef);
			case "ceil":
				return PMMLUtil.createApply(PMMLFunctions.CEIL, fieldRef);
			case "cos":
				return PMMLUtil.createApply(PMMLFunctions.COS, fieldRef);
			case "cosh":
				return PMMLUtil.createApply(PMMLFunctions.COSH, fieldRef);
			case "exp":
				return PMMLUtil.createApply(PMMLFunctions.EXP, fieldRef);
			case "expm1":
				return PMMLUtil.createApply(PMMLFunctions.EXPM1, fieldRef);
			case "floor":
				return PMMLUtil.createApply(PMMLFunctions.FLOOR, fieldRef);
			case "hypot":
				return PMMLUtil.createApply(PMMLFunctions.HYPOT, fieldRef);
			case "log":
				return PMMLUtil.createApply(PMMLFunctions.LN, fieldRef);
			case "log1p":
				return PMMLUtil.createApply(PMMLFunctions.LN1P, fieldRef);
			case "log10":
				return PMMLUtil.createApply(PMMLFunctions.LOG10, fieldRef);
			case "negative":
				return PMMLUtil.createApply(PMMLFunctions.MULTIPLY, PMMLUtil.createConstant(-1), fieldRef);
			case "reciprocal":
				return PMMLUtil.createApply(PMMLFunctions.DIVIDE, PMMLUtil.createConstant(1), fieldRef);
			case "rint":
				return PMMLUtil.createApply(PMMLFunctions.RINT, fieldRef);
			case "sign":
				return PMMLUtil.createApply(PMMLFunctions.IF, PMMLUtil.createApply(PMMLFunctions.LESSTHAN, fieldRef, PMMLUtil.createConstant(0)),
					PMMLUtil.createConstant(-1), // x < 0
					PMMLUtil.createApply(PMMLFunctions.IF, PMMLUtil.createApply(PMMLFunctions.GREATERTHAN, fieldRef, PMMLUtil.createConstant(0)),
						PMMLUtil.createConstant(+1), // x > 0
						PMMLUtil.createConstant(0) // x == 0
					)
				);
			case "sin":
				return PMMLUtil.createApply(PMMLFunctions.SIN, fieldRef);
			case "sinh":
				return PMMLUtil.createApply(PMMLFunctions.SINH, fieldRef);
			case "sqrt":
				return PMMLUtil.createApply(PMMLFunctions.SQRT, fieldRef);
			case "square":
				return PMMLUtil.createApply(PMMLFunctions.MULTIPLY, fieldRef, fieldRef);
			case "tan":
				return PMMLUtil.createApply(PMMLFunctions.TAN, fieldRef);
			case "tanh":
				return PMMLUtil.createApply(PMMLFunctions.TANH, fieldRef);
			default:
				throw new IllegalArgumentException(name);
		}
	}
}