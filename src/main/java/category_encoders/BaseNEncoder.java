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
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Strings;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.SetMultimap;
import org.jpmml.converter.BaseNFeature;
import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class BaseNEncoder extends CategoryEncoder {

	public BaseNEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Integer base = getBase();
		List<String> dropCols = null;
		Boolean dropInvariant = getDropInvariant();
		OrdinalEncoder ordinalEncoder = getOrdinalEncoder();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		List<Mapping> mappings = getMapping();

		if(base < 2 || base > 36){
			throw new IllegalArgumentException(Integer.toString(base));
		} // End if

		if(dropInvariant){
			dropCols = getDropCols();
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

		List<Mapping> ordinalMappings = ordinalEncoder.getMapping();

		ClassDictUtil.checkSize(mappings, ordinalMappings, features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Mapping mapping = mappings.get(i);

			Mapping ordinalMapping = ordinalMappings.get(i);

			Map<Object, Integer> ordinalCategoryMappings = OrdinalEncoder.getCategoryMapping(ordinalMapping);

			// XXX
			ordinalCategoryMappings.remove(CategoryEncoder.CATEGORY_MISSING);

			int requiredDigits = calcRequiredDigits(ordinalCategoryMappings, base);

			Map<Object, String> baseCategoryMappings = baseEncodeValues(ordinalCategoryMappings, base, requiredDigits);

			List<Feature> baseFeatures = new ArrayList<>();

			for(int pos = 0; pos < requiredDigits; pos++){
				String col = (String.valueOf(i) + "_" + String.valueOf(pos));

				if(dropCols != null && dropCols.contains(col)){
					continue;
				}

				SetMultimap<Integer, Object> values = LinkedHashMultimap.create();

				Collection<Map.Entry<Object, String>> entries = baseCategoryMappings.entrySet();
				for(Map.Entry<Object, String> entry : entries){
					Object category = entry.getKey();
					String baseValue = entry.getValue();

					char digit = baseValue.charAt(pos);

					values.put(Character.getNumericValue(digit), category);
				}

				baseFeatures.add(new BaseNFeature(encoder, feature, base, pos, values));
			}

			result.addAll(baseFeatures);
		}

		return result;
	}

	public Integer getBase(){
		return getInteger("base");
	}

	public OrdinalEncoder getOrdinalEncoder(){
		return get("ordinal_encoder", OrdinalEncoder.class);
	}

	static
	private int calcRequiredDigits(Map<?, Integer> ordinalCategoryMappings, int base){

		if(base == 1){
			return ordinalCategoryMappings.size() + 1;
		} else

		{
			return (int)Math.ceil(Math.log(ordinalCategoryMappings.size()) / Math.log(base)) + 1;
		}
	}

	static
	public Map<Object, String> baseEncodeValues(Map<Object, Integer> categoryMappings, int base, int requiredDigits){
		Map<Object, String> result = new LinkedHashMap<>();

		Collection<Map.Entry<Object, Integer>> entries = categoryMappings.entrySet();
		for(Map.Entry<Object, Integer> entry : entries){
			String baseValue = Strings.padStart(Integer.toString(entry.getValue(), base), requiredDigits, '0');

			result.put(entry.getKey(), baseValue);
		}

		return result;
	}
}