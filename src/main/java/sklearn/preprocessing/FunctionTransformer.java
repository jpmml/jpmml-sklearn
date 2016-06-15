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
import org.dmg.pmml.Apply;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
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

		String function = parseUFunc(ufunc);

		List<Feature> features = new ArrayList<>();

		for(int i = 0; i < inputFeatures.size(); i++){
			String id = ids.get(i);
			Feature inputFeature = inputFeatures.get(i);

			Apply apply = PMMLUtil.createApply(function, inputFeature.ref());

			DerivedField derivedField = featureMapper.createDerivedField(FieldName.create(function + "(" + id + ")"), apply);

			features.add(new ContinuousFeature(derivedField));
		}

		return features;
	}

	public Object getFunc(){
		return get("func");
	}

	static
	private String parseUFunc(UFunc ufunc){
		String module = ufunc.getModule();
		String name = ufunc.getName();

		switch(module){
			case "numpy":
			case "numpy.core.numeric":
			case "numpy.lib.function_base":
				switch(name){
					case "absolute":
						return "abs";
					case "exp":
						return name;
					case "log":
						return "ln";
					case "log10":
					case "sqrt":
						return name;
					default:
						throw new IllegalArgumentException(name);
				}
			default:
				throw new IllegalArgumentException(module);
		}
	}
}