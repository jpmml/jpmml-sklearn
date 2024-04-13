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
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.SkLearnTransformer;

public class PowerTransformer extends SkLearnTransformer {

	public PowerTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Number> lambdas = getLambdas();
		String method = getMethod();
		Boolean standardize = getStandardize();

		ClassDictUtil.checkSize(features, lambdas);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Number lambda = lambdas.get(i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Apply apply;

			if(!ValueUtil.isZero(lambda)){
				// "($name ^ lambda - 1) / lambda"
				apply = ExpressionUtil.createApply(PMMLFunctions.DIVIDE, ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, ExpressionUtil.createApply(PMMLFunctions.POW, continuousFeature.ref(), ExpressionUtil.createConstant(lambda)), ExpressionUtil.createConstant(1d)), ExpressionUtil.createConstant(lambda));
			} else

			{
				apply = ExpressionUtil.createApply(PMMLFunctions.LN, continuousFeature.ref());
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

	public List<Number> getLambdas(){
		return getNumberArray("lambdas_");
	}

	public String getMethod(){
		return getEnum("method", this::getString, Arrays.asList(PowerTransformer.METHOD_BOXCOX));
	}

	public StandardScaler getScaler(){
		return get("_scaler", StandardScaler.class);
	}

	public Boolean getStandardize(){
		return getBoolean("standardize");
	}

	private static final String METHOD_BOXCOX = "box-cox";
}