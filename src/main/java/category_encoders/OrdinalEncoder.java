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
import com.google.common.collect.Lists;
import numpy.DType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.python.PythonObject;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.Index;
import pandas.core.Series;
import pandas.core.SingleBlockManager;
import sklearn.Transformer;
import sklearn.preprocessing.EncoderUtil;

public class OrdinalEncoder extends Transformer {

	public OrdinalEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		return DataType.STRING;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		List<Mapping> mappings = getMapping();

		ClassDictUtil.checkSize(mappings, features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < mappings.size(); i++){
			Feature feature = features.get(i);
			Mapping mapping = mappings.get(i);

			SingleBlockManager mappingData = (mapping.getMapping()).getData();

			Index blockItems = mappingData.getOnlyBlockItem();

			List<Object> categories = new ArrayList<>();
			categories.addAll((blockItems.getData()).getData());

			HasArray blockValues = mappingData.getOnlyBlockValue();

			List<Integer> indices = new ArrayList<>();
			indices.addAll(ValueUtil.asIntegers((List)blockValues.getArrayContent()));

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

			result.add(EncoderUtil.encodeIndexFeature(feature, categories, indices, mapMissingTo, defaultValue, DataType.INTEGER, encoder));
		}

		return result;
	}

	public String getHandleMissing(){
		return getString("handle_missing");
	}

	public String getHandleUnknown(){
		return getString("handle_unknown");
	}

	public List<Mapping> getMapping(){
		List<Map<String, ?>> mapping = (List)getList("mapping", Map.class);

		Function<Map<String, ?>, Mapping> function = new Function<Map<String, ?>, Mapping>(){

			@Override
			public Mapping apply(Map<String, ?> map){
				Mapping mapping = OrdinalEncoder.this.new Mapping("mapping");
				mapping.putAll(map);

				return mapping;
			}
		};

		return Lists.transform(mapping, function);
	}

	private class Mapping extends PythonObject {

		private Mapping(String name){
			super(OrdinalEncoder.this.getPythonModule() + "." + OrdinalEncoder.this.getPythonName(), name);
		}

		public Integer getCol(){
			return getInteger("col");
		}

		public DType getDataType(){
			return get("data_type", DType.class);
		}

		public Series getMapping(){
			return get("mapping", Series.class);
		}
	}
}