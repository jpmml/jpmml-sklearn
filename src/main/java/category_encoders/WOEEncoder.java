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
package category_encoders;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import numpy.core.ScalarUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.Series;

public class WOEEncoder extends CategoryEncoder {

	public WOEEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Map<Integer, Series> mapping = getMapping();
		OrdinalEncoder ordinalEncoder = getOrdinalEncoder();

		List<OrdinalEncoder.Mapping> ordinalMappings = ordinalEncoder.getMapping();

		ClassDictUtil.checkSize(ordinalMappings, features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			OrdinalEncoder.Mapping ordinalMapping = ordinalMappings.get(i);

			Map<?, Integer> ordinalCategoryMappings = ordinalMapping.getCategoryMapping();

			// XXX
			ordinalCategoryMappings.remove(CategoryEncoder.CATEGORY_MISSING);

			Series series = mapping.get((Integer)i);

			Map<Integer, Double> woeMappings = CategoryEncoderUtil.toMap(series, key -> ValueUtil.asInt((Number)key), ValueUtil::asDouble);

			Map<?, Double> categoryScores = CategoryEncoderUtil.toCategoryValueMap(ordinalCategoryMappings, woeMappings);

			List<Object> categories = new ArrayList<>();
			categories.addAll(categoryScores.keySet());

			encoder.toCategorical(feature.getName(), categories);

			MapValues mapValues = PMMLUtil.createMapValues(feature.getName(), categoryScores);

			DerivedField derivedField = encoder.createDerivedField(FieldNameUtil.create("woe", feature), OpType.CATEGORICAL, DataType.DOUBLE, mapValues);

			result.add(new ThresholdFeature(encoder, derivedField, categoryScores));
		}

		return result;
	}

	public Map<Integer, Series> getMapping(){
		Map<?, ?> mapping = get("mapping", Map.class);

		return CategoryEncoderUtil.toTransformedMap(mapping, key -> ValueUtil.asInteger((Number)ScalarUtil.decode(key)), value -> (Series)value);
	}

	public OrdinalEncoder getOrdinalEncoder(){
		return get("ordinal_encoder", OrdinalEncoder.class);
	}
}