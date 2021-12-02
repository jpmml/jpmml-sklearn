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
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

import com.google.common.base.Functions;
import numpy.core.ScalarUtil;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.BlockManager;
import pandas.core.DataFrame;
import pandas.core.Index;
import pandas.core.Series;
import pandas.core.SeriesUtil;
import pandas.core.SingleBlockManager;
import sklearn.preprocessing.EncoderUtil;

abstract
public class MeanEncoder extends MapEncoder {

	public MeanEncoder(String module, String name){
		super(module, name);
	}

	abstract
	public MeanFunction createFunction();

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

		Object missingCategory = null;

		switch(handleMissing){
			case "error":
				break;
			case "value":
				missingCategory = CategoryEncoder.CATEGORY_NAN;
				break;
			default:
				throw new IllegalArgumentException(handleMissing);
		}

		Number defaultValue = null;

		switch(handleUnknown){
			case "error":
				break;
			case "value":
				defaultValue = getMean();
				break;
			default:
				throw new IllegalArgumentException(handleUnknown);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Object col = cols.get(i);

			Series series = mapping.get(col);

			Map<Object, Double> categoryMeans = SeriesUtil.toMap(series, Functions.identity(), ValueUtil::asDouble);

			List<Object> categories = new ArrayList<>(categoryMeans.keySet());

			Field<?> field = encoder.toCategorical(feature.getName(), EncoderUtil.filterCategories(categories));

			switch(handleUnknown){
				case "value":
					{
						EncoderUtil.addDecorator(field, new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_IS, null), encoder);
					}
					break;
				default:
					break;
			}

			Feature mapFeature = new MapFeature(encoder, feature, categoryMeans, missingCategory, defaultValue){

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
		Map<?, ?> mapping = get("mapping", Map.class);

		return CategoryEncoderUtil.toTransformedMap(mapping, key -> ScalarUtil.decode(key), value -> toMeanSeries((DataFrame)value, createFunction()));
	}

	public Double getMean(){
		return ValueUtil.asDouble(getNumber("_mean"));
	}

	static
	private Series toMeanSeries(DataFrame dataFrame, MeanFunction function){
		BlockManager blockManager = dataFrame.getData();

		List<Index> axes = blockManager.getAxesArray();
		if(axes.size() != 2){
			throw new IllegalArgumentException();
		}

		List<?> firstDim = (axes.get(0)).getDataData();
		List<?> secondDim = (axes.get(1)).getDataData();

		if(!(Arrays.asList("sum", "count")).equals(firstDim)){
			throw new IllegalArgumentException();
		}

		List<HasArray> blockValues = blockManager.getBlockValues();
		if(blockValues.size() != 2){
			throw new IllegalArgumentException();
		}

		List<?> sumValues = (blockValues.get(0)).getArrayContent();
		List<?> countValues = (blockValues.get(1)).getArrayContent();

		List<Double> meanValues = new ArrayList<>();

		for(int i = 0; i < sumValues.size(); i++){
			Double sum = ValueUtil.asDouble((Number)sumValues.get(i));
			Integer count = ValueUtil.asInteger((Number)countValues.get(i));

			Double mean = function.apply(sum, count);

			meanValues.add(mean);
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

		SingleBlockManager singleBlockManager = new SingleBlockManager();
		singleBlockManager.setOnlyBlockItem(axes.get(1));
		singleBlockManager.setOnlyBlockValue(hasArray);

		Series result = new Series();
		result.setBlockManager(singleBlockManager);

		return result;
	}

	public interface MeanFunction extends BiFunction<Double, Integer, Double> {

		@Override
		Double apply(Double sum, Integer count);
	}
}