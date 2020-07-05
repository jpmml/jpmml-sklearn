/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class EncoderUtil {

	private EncoderUtil(){
	}

	static
	public Feature encodeIndexFeature(Transformer transformer, Feature feature, List<?> categories, DataType dataType, SkLearnEncoder encoder){
		List<Number> indexCategories = new ArrayList<>(categories.size());

		for(int i = 0; i < categories.size(); i++){

			switch(dataType){
				case INTEGER:
					indexCategories.add(i);
					break;
				case FLOAT:
					indexCategories.add((float)i);
					break;
				case DOUBLE:
					indexCategories.add((double)i);
					break;
				default:
					throw new IllegalArgumentException();
			}
		}

		return encodeIndexFeature(transformer, feature, categories, indexCategories, null, null, dataType, encoder);
	}

	static
	public Feature encodeIndexFeature(Transformer transformer, Feature feature, List<?> categories, List<? extends Number> indexCategories, Number mapMissingTo, Number defaultValue, DataType dataType, SkLearnEncoder encoder){
		ClassDictUtil.checkSize(categories, indexCategories);

		encoder.toCategorical(feature.getName(), categories);

		MapValues mapValues = PMMLUtil.createMapValues(feature.getName(), categories, indexCategories)
			.setMapMissingTo(mapMissingTo)
			.setDefaultValue(defaultValue);

		DerivedField derivedField = encoder.createDerivedField(transformer.createFieldName("encoder", feature), OpType.CATEGORICAL, dataType, mapValues);

		Feature encodedFeature = new IndexFeature(encoder, derivedField, indexCategories);

		Feature result = new CategoricalFeature(encoder, feature, categories){

			@Override
			public ContinuousFeature toContinuousFeature(){
				return encodedFeature.toContinuousFeature();
			}
		};

		return result;
	}

	static
	public List<List<?>> transformCategories(List<HasArray> arrays){
		Function<HasArray, List<?>> function = new Function<HasArray, List<?>>(){

			@Override
			public List<?> apply(HasArray hasArray){
				return hasArray.getArrayContent();
			}
		};

		return Lists.transform(arrays, function);
	}
}