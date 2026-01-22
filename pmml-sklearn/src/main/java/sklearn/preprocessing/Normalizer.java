/*
 * Copyright (c) 2026 Villu Ruusmann
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
import java.util.function.Function;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.SkLearnTransformer;

public class Normalizer extends SkLearnTransformer {

	public Normalizer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String norm = getNorm();

		Apply normApply = encodeNorm(norm, features);

		DerivedField normDerivedField = encoder.createDerivedField(FieldNameUtil.create("norm", norm), OpType.CONTINUOUS, DataType.DOUBLE, normApply);

		Feature normFeature = new ContinuousFeature(encoder, normDerivedField);

		List<Feature> result = new ArrayList<>();

		for(Feature feature : features){
			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Apply apply = ExpressionUtil.createApply(PMMLFunctions.DIVIDE, continuousFeature.ref(), normFeature.ref());

			DerivedField derivedField = encoder.createDerivedField(createFieldName("normalizer", continuousFeature), OpType.CONTINUOUS, DataType.DOUBLE, apply);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	private Apply encodeNorm(String norm, List<? extends Feature> features){

		switch(norm){
			case NORM_L1:
				return encodeAggregation(
					PMMLFunctions.SUM,
					(feature) -> {
						ContinuousFeature continuousFeature = feature.toContinuousFeature();

						return ExpressionUtil.createApply(PMMLFunctions.ABS, continuousFeature.ref());
					},
					features
				);
			case NORM_L2:
				return ExpressionUtil.createApply(
					PMMLFunctions.SQRT,
					encodeAggregation(
						PMMLFunctions.SUM,
						(feature) -> {
							ContinuousFeature continuousFeature = feature.toContinuousFeature();

							return ExpressionUtil.createApply(PMMLFunctions.POW, continuousFeature.ref(), ExpressionUtil.createConstant(2d));
						},
						features
					)
				);
			case NORM_MAX:
				return encodeAggregation(
					PMMLFunctions.MAX,
					(feature) -> {
						ContinuousFeature continuousFeature = feature.toContinuousFeature();

						return ExpressionUtil.createApply(PMMLFunctions.ABS, continuousFeature.ref());
					},
					features
				);
			default:
				throw new IllegalArgumentException(norm);
		}
	}

	public String getNorm(){
		return getEnum("norm", this::getString, Arrays.asList(Normalizer.NORM_L1, Normalizer.NORM_L2, Normalizer.NORM_MAX));
	}

	static
	private Apply encodeAggregation(String function, Function<Feature, Apply> featureEncoder, List<? extends Feature> features){
		Apply apply = ExpressionUtil.createApply(function);

		for(Feature feature : features){
			apply.addExpressions(featureEncoder.apply(feature));
		}

		return apply;
	}

	private static final String NORM_L1 = "l1";
	private static final String NORM_L2 = "l2";
	private static final String NORM_MAX = "max";
}