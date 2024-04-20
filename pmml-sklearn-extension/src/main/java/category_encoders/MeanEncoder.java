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
import java.util.Objects;
import java.util.function.BiFunction;

import com.google.common.base.Functions;
import com.google.common.collect.Iterables;
import numpy.core.ScalarUtil;
import org.dmg.pmml.Field;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.jpmml.converter.CMatrixUtil;
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
		@SuppressWarnings("unused")
		Boolean dropInvariant = getDropInvariant();
		String handleMissing = getHandleMissing();
		String handleUnknown = getHandleUnknown();
		Map<Object, Series> mapping = getMapping();

		Object missingCategory = null;

		if(Objects.equals(MeanEncoder.HANDLEMISSING_VALUE, handleMissing)){
			missingCategory = BaseEncoder.CATEGORY_NAN;
		}

		Number defaultValue = null;

		if(Objects.equals(MeanEncoder.HANDLEUNKNOWN_VALUE, handleUnknown)){
			defaultValue = getMean();
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
				case MeanEncoder.HANDLEUNKNOWN_VALUE:
					{
						EncoderUtil.addDecorator(field, new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_IS, null), encoder);
					}
					break;
				default:
					break;
			}

			Feature mapFeature = new MapFeature(encoder, feature, categoryMeans, missingCategory, defaultValue){

				@Override
				public String getDerivedName(){
					return createFieldName(functionName(), getName());
				}
			};

			result.add(mapFeature);
		}

		return result;
	}

	@Override
	public Map<Object, Series> getMapping(){
		Map<?, ?> mapping = getDict("mapping");

		return CategoryEncoderUtil.toTransformedMap(mapping, key -> ScalarUtil.decode(key), value -> toMeanSeries((DataFrame)value, createFunction()));
	}

	public Double getMean(){
		return ValueUtil.asDouble(getNumber("_mean"));
	}

	static
	private Series toMeanSeries(DataFrame dataFrame, MeanFunction function){
		BlockManager data = dataFrame.getData();

		Index columnAxis = data.getColumnAxis();
		Index rowAxis = data.getRowAxis();

		if(!(Arrays.asList("sum", "count")).equals(columnAxis.getValues())){
			throw new IllegalArgumentException();
		}

		List<HasArray> blockValues = data.getBlockValues();

		List<?> sumValues;
		List<?> countValues;

		// CategoryEncoders 2.3
		if(blockValues.size() == 2){
			sumValues = (blockValues.get(0)).getArrayContent();
			countValues = (blockValues.get(1)).getArrayContent();
		} else

		// CategoryEncoders 2.5+
		{
			HasArray blockValue = Iterables.getOnlyElement(blockValues);

			List<?> blockValueContent = blockValue.getArrayContent();
			int[] blockValueShape = blockValue.getArrayShape();

			sumValues = CMatrixUtil.getRow(blockValueContent, blockValueShape[0], blockValueShape[1], 0);
			countValues = CMatrixUtil.getRow(blockValueContent, blockValueShape[0], blockValueShape[1], 1);
		}

		List<Double> meanValues = new ArrayList<>();

		for(int i = 0; i < sumValues.size(); i++){
			Double sum = ValueUtil.asDouble((Number)sumValues.get(i));
			Integer count = ValueUtil.asInteger((Number)countValues.get(i));

			Double mean = function.apply(sum, count);

			meanValues.add(mean);
		}

		return SeriesUtil.createSeries(rowAxis, meanValues);
	}

	public interface MeanFunction extends BiFunction<Double, Integer, Double> {

		@Override
		Double apply(Double sum, Integer count);
	}
}