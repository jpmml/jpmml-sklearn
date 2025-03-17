/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn2pmml.preprocessing;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;

public class AggregateTransformer extends Transformer {

	public AggregateTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String function = getFunction();

		if(features.size() <= 1){
			return features;
		}

		Apply apply = ExpressionUtil.createApply(translateFunction(function));

		for(Feature feature : features){
			apply.addExpressions(feature.ref());
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName(function, features), OpType.CONTINUOUS, DataType.DOUBLE, apply);

		return Collections.singletonList(new ContinuousFeature(encoder, derivedField));
	}

	public String getFunction(){
		return getEnum("function", this::getString, Arrays.asList(AggregateTransformer.FUNCTION_AVG, AggregateTransformer.FUNCTION_MAX, AggregateTransformer.FUNCTION_MEAN, AggregateTransformer.FUNCTION_MIN, AggregateTransformer.FUNCTION_PROD, AggregateTransformer.FUNCTION_PRODUCT, AggregateTransformer.FUNCTION_SUM));
	}

	static
	private String translateFunction(String function){

		switch(function){
			case AggregateTransformer.FUNCTION_AVG: // PMML-style
			case AggregateTransformer.FUNCTION_MEAN: // Python-style
				return PMMLFunctions.AVG;
			case AggregateTransformer.FUNCTION_MAX:
				return PMMLFunctions.MAX;
			case AggregateTransformer.FUNCTION_MIN:
				return PMMLFunctions.MIN;
			case AggregateTransformer.FUNCTION_PROD: // Python-style
			case AggregateTransformer.FUNCTION_PRODUCT: // PMML-style
				return PMMLFunctions.PRODUCT;
			case AggregateTransformer.FUNCTION_SUM:
				return PMMLFunctions.SUM;
			default:
				throw new IllegalArgumentException(function);
		}
	}

	private static final String FUNCTION_AVG = "avg";
	private static final String FUNCTION_MAX = "max";
	private static final String FUNCTION_MEAN = "mean";
	private static final String FUNCTION_MIN = "min";
	private static final String FUNCTION_PROD = "prod";
	private static final String FUNCTION_PRODUCT = "product";
	private static final String FUNCTION_SUM = "sum";
}