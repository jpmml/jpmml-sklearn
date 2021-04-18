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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.base.Functions;
import numpy.core.ScalarUtil;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.BlockManager;
import pandas.core.DataFrame;
import pandas.core.Index;
import pandas.core.Series;
import pandas.core.SingleBlockManager;

public class LeaveOneOutEncoder extends MapEncoder {

	public LeaveOneOutEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public String functionName(){
		return "loo";
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<?> cols = getCols();
		Boolean dropInvariant = getDropInvariant();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		Map<Object, Series> mapping = getMapping();

		if(dropInvariant){
			throw new IllegalArgumentException();
		}

		switch(handleMissing){
			case "error":
				break;
			default:
				throw new IllegalArgumentException(handleMissing);
		}

		switch(handleUnknown){
			case "error":
				break;
			default:
				throw new IllegalArgumentException(handleUnknown);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Object col = cols.get(i);

			Series series = mapping.get(col);

			Map<Object, Number> categoryMeans = CategoryEncoderUtil.toMap(series, Functions.identity(), ValueUtil::asDouble);

			List<Object> categories = new ArrayList<>();
			categories.addAll(categoryMeans.keySet());

			encoder.toCategorical(feature.getName(), categories);

			Feature mapFeature = new MapFeature(encoder, feature, categoryMeans){

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
	public Map<Object, Series> getMapping(){
		Number mean = getMean();
		Map<?, ?> mapping = get("mapping", Map.class);

		return CategoryEncoderUtil.toTransformedMap(mapping, key -> ScalarUtil.decode(key), value -> toMeanSeries((DataFrame)value, mean));
	}

	public Number getMean(){
		return getNumber("_mean");
	}

	static
	private Series toMeanSeries(DataFrame dataFrame, Number mean){
		BlockManager blockManager = dataFrame.get("_mgr", BlockManager.class);

		List<Index> axes = blockManager.getAxesArray();
		if(axes.size() != 2){
			throw new IllegalArgumentException();
		}

		List<?> firstDim = (axes.get(0)).getData().getData();
		List<?> secondDim = (axes.get(1)).getData().getData();

		if(!(Arrays.asList("sum", "count")).equals(firstDim)){
			throw new IllegalArgumentException();
		}

		List<HasArray> blockValues = blockManager.getBlockValues();
		if(blockValues.size() != 2){
			throw new IllegalArgumentException();
		}

		List<?> sumValues = (blockValues.get(0)).getArrayContent();
		List<?> countValues = (blockValues.get(1)).getArrayContent();

		List<Number> meanValues = new ArrayList<>();

		for(int i = 0; i < sumValues.size(); i++){
			Number sum = (Number)sumValues.get(i);
			Number count = (Number)countValues.get(i);

			if(count.doubleValue() > 1d){
				meanValues.add(sum.doubleValue() / count.doubleValue());
			} else

			{
				meanValues.add(mean);
			}
		}

		HasArray hasArray = new HasArray(){

			@Override
			public List<?> getArrayContent(){
				return meanValues;
			}

			@Override
			public int[] getArrayShape(){
				return new int[]{meanValues.size()};
			}
		};

		SingleBlockManager singleBlockManager = new SingleBlockManager("pandas.core.internals.managers", "SingleBlockManager");
		singleBlockManager.put("block_items", Collections.singletonList(axes.get(1)));
		singleBlockManager.put("block_values", Collections.singletonList(hasArray));

		Series result = new Series("pandas.core.series", "Series");
		result.put("_data", singleBlockManager);

		return result;
	}
}