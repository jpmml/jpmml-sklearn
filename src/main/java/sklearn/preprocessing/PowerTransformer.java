/*
 * Copyright (c) 2021 Villu Ruusmann
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

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class PowerTransformer extends Transformer {

	public PowerTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<? extends Number> lambdas = getLambdas();
		String method = getMethod();
		Boolean standardize = getStandardize();

		switch(method){
			case "box-cox":
				break;
			default:
				throw new IllegalArgumentException(method);
		}

		ClassDictUtil.checkSize(features, lambdas);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Number lambda = lambdas.get(i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Apply apply;

			if(!ValueUtil.isZero(lambda)){
				apply = PMMLUtil.createApply(PMMLFunctions.DIVIDE)
					.addExpressions(
						PMMLUtil.createApply(PMMLFunctions.SUBTRACT, PMMLUtil.createApply(PMMLFunctions.POW, continuousFeature.ref(), PMMLUtil.createConstant(lambda)), PMMLUtil.createConstant(1d)),
						PMMLUtil.createConstant(lambda)
					);
			} else

			{
				apply = PMMLUtil.createApply(PMMLFunctions.LN, continuousFeature.ref());
			}

			DerivedField derivedField = encoder.createDerivedField(createFieldName("boxCox", continuousFeature.getName()), OpType.CONTINUOUS, DataType.DOUBLE, apply);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		if(standardize){
			StandardScaler scaler = getScaler();

			result = scaler.encode(result, encoder);
		}

		return result;
	}

	public List<? extends Number> getLambdas(){
		return getArray("lambdas_", Number.class);
	}

	public String getMethod(){
		return getString("method");
	}

	public StandardScaler getScaler(){
		return get("_scaler", StandardScaler.class);
	}

	public Boolean getStandardize(){
		return getBoolean("standardize");
	}
}