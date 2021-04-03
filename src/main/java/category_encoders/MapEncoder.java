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
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import numpy.core.ScalarUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.Series;

abstract
public class MapEncoder extends CategoryEncoder {

	public MapEncoder(String module, String name){
		super(module, name);
	}

	abstract
	public String functionName();

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean dropInvariant = getDropInvariant();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		Map<Integer, Series> mapping = getMapping();
		OrdinalEncoder ordinalEncoder = getOrdinalEncoder();

		if(dropInvariant){
			throw new IllegalArgumentException();
		}

		switch(handleMissing){
			case "value":
				break;
			default:
				throw new IllegalArgumentException(handleMissing);
		} // End switch

		switch(handleUnknown){
			case "value":
				break;
			default:
				throw new IllegalArgumentException(handleUnknown);
		}

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

			Map<Integer, Double> valueMappings = CategoryEncoderUtil.toMap(series, key -> ValueUtil.asInt((Number)key), ValueUtil::asDouble);

			Map<?, Double> categoryValues = mapEncodeValues(ordinalCategoryMappings, valueMappings);

			List<Object> categories = new ArrayList<>();
			categories.addAll(categoryValues.keySet());

			encoder.toCategorical(feature.getName(), categories);

			MapValues mapValues = PMMLUtil.createMapValues(feature.getName(), categoryValues);

			DerivedField derivedField = encoder.createDerivedField(createFieldName(functionName(), feature), OpType.CATEGORICAL, DataType.DOUBLE, mapValues);

			result.add(new ThresholdFeature(encoder, derivedField, categoryValues));
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

	static
	public <K, V> Map<K, V> mapEncodeValues(Map<K, Integer> categoryMappings, Map<Integer, V> valueMappings){
		Map<K, V> result = new LinkedHashMap<>();

		Collection<? extends Map.Entry<K, Integer>> entries = categoryMappings.entrySet();
		for(Map.Entry<K, Integer> entry : entries){
			K category = entry.getKey();
			V value = valueMappings.get(entry.getValue());

			if(value == null){
				throw new IllegalArgumentException();
			}

			result.put(category, value);
		}

		return result;
	}
}