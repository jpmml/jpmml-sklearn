/*
 * Copyright (c) 2024 Villu Ruusmann
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

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class TargetEncoder extends BaseEncoder {

	public TargetEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<List<Object>> categories = getCategories();
		List<List<Number>> encodings = getEncodings();
		Number targetMean = getTargetMean();
		String targetType = getTargetType();

		switch(targetType){
			case "continuous":
			case "binary":
				break;
			default:
				throw new IllegalArgumentException(targetType);
		}

		ClassDictUtil.checkSize(features.size(), categories, encodings);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			List<Object> featureCategories = categories.get(i);
			List<Number> featureEncodings = encodings.get(i);

			ClassDictUtil.checkSize(featureCategories, featureEncodings);

			// A NaN value or null
			Object missingCategory = getMissingCategory(featureCategories);

			Number mapMissingTo = null;

			int index = featureCategories.indexOf(missingCategory);
			if(index > -1){
				featureCategories = new ArrayList<>(featureCategories);
				featureCategories.remove(index);

				featureEncodings = new ArrayList<>(featureEncodings);
				mapMissingTo = featureEncodings.remove(index);
			}

			encoder.toCategorical(feature.getName(), featureCategories);

			MapValues mapValues = ExpressionUtil.createMapValues(feature.getName(), featureCategories, featureEncodings)
				.setMapMissingTo(mapMissingTo)
				.setDefaultValue(targetMean);

			DerivedField derivedField = encoder.createDerivedField(createFieldName("targetEncoder", feature), OpType.CONTINUOUS, DataType.DOUBLE, mapValues);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public List<List<Number>> getEncodings(){
		return getArrayList("encodings_", Number.class);
	}

	public Number getTargetMean(){
		return getNumber("target_mean_");
	}

	public String getTargetType(){
		return getString("target_type_");
	}

	static
	private Object getMissingCategory(List<?> categories){

		for(Object category : categories){

			if(ValueUtil.isNaN(category)){
				return category;
			}
		}

		return null;
	}
}