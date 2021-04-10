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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Functions;
import com.google.common.collect.Iterables;
import numpy.core.ScalarUtil;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.Series;

public class CountEncoder extends MapEncoder {

	public CountEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public String functionName(){
		return "count";
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<?> cols = getCols();
		Boolean dropInvariant = getDropInvariant();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		Boolean normalize = getNormalize();
		Map<Object, Series> mapping = getMapping();
		Map<Object, Map<Object, String>> minGroupCategories = getMinGroupCategories();

		if(dropInvariant){
			throw new IllegalArgumentException();
		}

		switch(handleMissing){
			case "error":
				break;
			default:
				throw new IllegalArgumentException(handleMissing);
		}

		// XXX
		if(handleUnknown == null){
			throw new IllegalArgumentException();
		}

		switch(handleUnknown){
			case "error":
				break;
			default:
				throw new IllegalArgumentException(handleUnknown);
		}

		ClassDictUtil.checkSize(features, cols);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Object col = cols.get(i);

			Series series = mapping.get(col);

			Map<Object, Number> categoryCounts = CategoryEncoderUtil.toMap(series, Functions.identity(), normalize ? ValueUtil::asDouble : ValueUtil::asInteger);

			Map<?, String> leftoverCategories = minGroupCategories.get(col);
			if(leftoverCategories != null){
				String leftoverCategory = Iterables.getOnlyElement(new HashSet<>(leftoverCategories.values()));

				Number leftoverCount = categoryCounts.remove(leftoverCategory);
				if(leftoverCount == null){
					throw new IllegalArgumentException();
				}

				Set<?> categories = leftoverCategories.keySet();
				for(Object category : categories){
					categoryCounts.put(category, leftoverCount);
				}
			}

			List<Object> categories = new ArrayList<>();
			categories.addAll(categoryCounts.keySet());

			encoder.toCategorical(feature.getName(), categories);

			Feature mapFeature = new MapFeature(encoder, feature, categoryCounts){

				@Override
				public FieldName getDerivedName(){
					return createFieldName(functionName(), getName());
				}
			};

			result.add(mapFeature);
		}

		return result;
	}

	@Override
	public String getHandleUnknown(){
		return getOptionalString("handle_unknown");
	}

	public Boolean getNormalize(){
		return getBoolean("normalize");
	}

	public Map<Object, Map<Object, String>> getMinGroupCategories(){
		Map<?, ?> minGroupCategories = get("_min_group_categories", Map.class);

		return CategoryEncoderUtil.toTransformedMap(minGroupCategories, key -> ScalarUtil.decode(key), value -> (Map)value);
	}
}