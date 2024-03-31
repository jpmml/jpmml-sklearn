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
import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonObject;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.Series;
import pandas.core.SeriesUtil;
import sklearn.preprocessing.EncoderUtil;

public class OrdinalEncoder extends BaseEncoder {

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

		switch(handleMissing){
			case "error":
			case "return_nan":
			case "value":
				break;
			default:
				throw new IllegalArgumentException(handleMissing);
		} // End switch

		switch(handleUnknown){
			case "error":
			case "value":
				break;
			default:
				throw new IllegalArgumentException(handleUnknown);
		}

		ClassDictUtil.checkSize(features, mappings);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Mapping mapping = mappings.get(i);

			Map<?, Integer> categoryMappings = mapping.getCategoryMapping();

			List<Object> categories = new ArrayList<>();
			categories.addAll(categoryMappings.keySet());

			List<Integer> indices = new ArrayList<>();
			indices.addAll(categoryMappings.values());

			Integer defaultValue = null;

			switch(handleUnknown){
				case "value":
					defaultValue = -2;
					break;
				default:
					break;
			}

			result.add(EncoderUtil.encodeIndexFeature(this, feature, categories, indices, null, defaultValue, DataType.INTEGER, encoder));
		}

		return result;
	}

	@SuppressWarnings({"rawtypes", "unchecked"})
	public List<Mapping> getMapping(){
		List<Map<String, ?>> mapping = (List)getList("mapping", Map.class);

		Function<Map<String, ?>, Mapping> function = new Function<Map<String, ?>, Mapping>(){

			@Override
			public Mapping apply(Map<String, ?> map){
				Mapping mapping = new Mapping(getClassName(), "mapping");
				mapping.putAll(map);

				return mapping;
			}
		};

		return Lists.transform(mapping, function);
	}

	static
	public class Mapping extends PythonObject {

		private Mapping(String module, String name){
			super(module, name);
		}

		public Map<?, Integer> getCategoryMapping(){

			// XXX
			try {
				Series mapping = get("mapping", Series.class);

				return SeriesUtil.toMap(mapping, Functions.identity(), ValueUtil::asInteger);
			} catch(IllegalArgumentException iae){
				return (Map)getDict("mapping");
			}
		}
	}

	public static final Object CATEGORY_UNKNOWN = -1;
}