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
import org.dmg.pmml.Field;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.Series;
import pandas.core.SeriesUtil;
import sklearn.preprocessing.EncoderUtil;

abstract
public class OrdinalMapEncoder extends MapEncoder {

	public OrdinalMapEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<?> cols = getCols();
		Boolean dropInvariant = getDropInvariant();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		Map<Object, Series> mapping = getMapping();
		OrdinalEncoder ordinalEncoder = getOrdinalEncoder();

		if(dropInvariant){
			throw new IllegalArgumentException();
		}

		Object missingCategory = null;

		if((OrdinalMapEncoder.HANDLEMISSING_VALUE).equals(handleMissing)){
			missingCategory = BaseEncoder.CATEGORY_NAN;
		}

		Object unknownCategory = null;

		if((OrdinalMapEncoder.HANDLEUNKNOWN_VALUE).equals(handleUnknown)){
			unknownCategory = OrdinalEncoder.CATEGORY_UNKNOWN;
		}

		List<OrdinalEncoder.Mapping> ordinalMappings = ordinalEncoder.getMapping();

		ClassDictUtil.checkSize(features, cols, ordinalMappings);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Object col = ScalarUtil.decode(cols.get(i));
			OrdinalEncoder.Mapping ordinalMapping = ordinalMappings.get(i);

			Map<?, Integer> ordinalCategoryMappings = ordinalMapping.getCategoryMapping();

			switch(handleMissing){
				case OrdinalMapEncoder.HANDLEMISSING_ERROR:
					ordinalCategoryMappings.remove(BaseEncoder.CATEGORY_NAN);
					break;
				default:
					break;
			}

			Series series = mapping.get(col);
			if(series == null){
				throw new IllegalArgumentException(String.valueOf(col));
			}

			Map<Integer, Double> valueMappings = SeriesUtil.toMap(series, key -> ValueUtil.asInteger((Number)key), ValueUtil::asDouble);

			Map<?, Double> categoryValues = mapEncodeValues(ordinalCategoryMappings, valueMappings);

			List<Object> categories = new ArrayList<>(categoryValues.keySet());

			Field<?> field = encoder.toCategorical(feature.getName(), EncoderUtil.filterCategories(categories));

			Double defaultValue = null;

			switch(handleUnknown){
				case OrdinalMapEncoder.HANDLEUNKNOWN_VALUE:
					{
						if(!valueMappings.containsKey(unknownCategory)){
							throw new IllegalArgumentException();
						}

						defaultValue = valueMappings.get(unknownCategory);

						EncoderUtil.addDecorator(field, new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_IS, null), encoder);
					}
					break;
				default:
					break;
			}

			Feature mapFeature = new MapFeature(encoder, feature, categoryValues, missingCategory, defaultValue){

				@Override
				public String getDerivedName(){
					return createFieldName(functionName(), getName());
				}
			};

			result.add(mapFeature);
		}

		return result;
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