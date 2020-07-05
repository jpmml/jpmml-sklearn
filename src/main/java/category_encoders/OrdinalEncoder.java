/*
 * Copyright (c) 2020 Villu Ruusmann
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.Index;
import pandas.core.Series;
import pandas.core.SingleBlockManager;
import sklearn.preprocessing.EncoderUtil;

public class OrdinalEncoder extends CategoryEncoder {

	public OrdinalEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean dropInvariant = getDropInvariant();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		List<Mapping> mappings = getMapping();

		if(dropInvariant){
			throw new IllegalArgumentException();
		}

		ClassDictUtil.checkSize(mappings, features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Mapping mapping = mappings.get(i);

			Map<Object, Integer> categoryMappings = getCategoryMapping(mapping);

			List<Object> categories = new ArrayList<>();
			categories.addAll(categoryMappings.keySet());

			List<Integer> indices = new ArrayList<>();
			indices.addAll(categoryMappings.values());

			Number mapMissingTo = null;

			switch(handleMissing){
				case "value":
					{
						Number lastCategory = (Number)categories.get(categories.size() - 1);
						if(!Double.isNaN(lastCategory.doubleValue())){
							throw new IllegalArgumentException(String.valueOf(lastCategory));
						}

						Integer lastIndex = indices.get(indices.size() - 1);
						if(lastIndex != -2){
							throw new IllegalArgumentException(String.valueOf(lastIndex));
						}

						categories = categories.subList(0, categories.size() - 1);
						indices = indices.subList(0, indices.size() - 1);

						mapMissingTo = -2;
					}
					break;
				default:
					throw new IllegalArgumentException(handleMissing);
			}

			Number defaultValue = null;

			switch(handleUnknown){
				case "value":
					{
						defaultValue = -1;
					}
					break;
				default:
					throw new IllegalArgumentException(handleUnknown);
			}

			result.add(EncoderUtil.encodeIndexFeature(this, feature, categories, indices, mapMissingTo, defaultValue, DataType.INTEGER, encoder));
		}

		return result;
	}

	static
	public Map<Object, Integer> getCategoryMapping(Mapping mapping){
		SingleBlockManager mappingData = (mapping.getMapping(Series.class)).getData();

		Index blockItem = mappingData.getOnlyBlockItem();
		List<?> categories = (blockItem.getData()).getData();

		HasArray blockValue = mappingData.getOnlyBlockValue();
		List<Integer> indices = ValueUtil.asIntegers((List)blockValue.getArrayContent());

		ClassDictUtil.checkSize(categories, indices);

		Map<Object, Integer> result = new LinkedHashMap<>();

		for(int i = 0; i < categories.size(); i++){
			result.put(categories.get(i), indices.get(i));
		}

		return result;
	}
}