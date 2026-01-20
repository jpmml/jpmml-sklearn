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
import java.util.Collections;
import java.util.List;

import net.razorvine.pickle.objects.ClassDictConstructor;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.python.AttributeException;
import org.jpmml.python.ClassDictConstructorUtil;
import org.jpmml.python.FunctionUtil;
import org.jpmml.python.Identifiable;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;
import sklearn.IdentifiableUtil;
import sklearn.SkLearnTransformer;
import sklearn2pmml.preprocessing.ExpressionTransformer;

public class FunctionTransformer extends SkLearnTransformer {

	public FunctionTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Identifiable func = getFunc();

		if(func == null){
			return features;
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Expression expression = FunctionUtil.encodeFunction(func, Collections.singletonList(continuousFeature.ref()), encoder);

			DerivedField derivedField = encoder.createDerivedField(createFieldName(func.getName(), continuousFeature), OpType.CONTINUOUS, DataType.DOUBLE, expression);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public Identifiable getFunc(){
		Identifiable func;

		try {
			func = getOptionalIdentifiable("func");
		} catch(AttributeException ae){
			String name = getFunctionName("func");

			if(name != null && name.startsWith("__main__" + ".")){
				name = name.substring(("__main__" + ".").length());

				throw new SkLearnException("The function " + ExceptionUtil.formatName(name) + " does not have a persistent state", ae)
					.setSolution("Use the " + (ExpressionTransformer.class).getName() + " transformer to give this function a persistent state")
					.setExample(ExpressionTransformer.formatComplexExample(name));
			}

			throw ae;
		}

		return IdentifiableUtil.filter(func);
	}

	public Identifiable getInverseFunc(){
		Identifiable inverseFunc = getOptionalIdentifiable("inverse_func");

		return IdentifiableUtil.filter(inverseFunc);
	}

	private String getFunctionName(String name){
		Object object = getOptionalObject(name);

		if(object instanceof ClassDictConstructor){
			ClassDictConstructor dictConstructor = (ClassDictConstructor)object;

			return ClassDictConstructorUtil.getClassName(dictConstructor);
		}

		return null;
	}
}