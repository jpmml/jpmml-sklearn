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
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.BinaryThresholdFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Decorator;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueFeature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class MultiOneHotEncoder extends BaseEncoder {

	public MultiOneHotEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<List<Object>> categories = getCategories();
		Object drop = getDrop();
		List<Integer> dropIdx = (drop != null ? getDropIdx() : null);
		String handleUnknown = getHandleUnknown();
		Boolean infrequentEnabled = getInfrequentEnabled();
		List<List<Number>> infrequentIndices = (infrequentEnabled ? getInfrequentIndices() : null);

		ClassDictUtil.checkSize(categories, features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			List<Object> featureCategories = new ArrayList<>(categories.get(i));
			List<Number> featureInfrequentIndices = (infrequentEnabled ? infrequentIndices.get(i) : null);

			boolean featureInfrequentEnabled = infrequentEnabled;
			if(featureInfrequentIndices == null || featureInfrequentIndices.isEmpty()){
				featureInfrequentEnabled = false;
			}

			Object infrequentCategory = null;

			if(featureInfrequentEnabled){
				infrequentCategory = getInfrequentCategory(feature);
			}

			Decorator invalidValueDecorator;

			switch(handleUnknown){
				case "error":
					{
						invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.RETURN_INVALID, null);
					}
					break;
				case "ignore":
					{
						invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_IS, null);
					}
					break;
				case "infrequent_if_exist":
					{
						if(featureInfrequentEnabled){
							invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_VALUE, infrequentCategory);
						} else

						{
							invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_IS, null);
						}
					}
					break;
				default:
					throw new IllegalArgumentException(handleUnknown);
			}

			EncoderUtil.addDecorator(feature, invalidValueDecorator);

			if(feature instanceof BinaryThresholdFeature){
				BinaryThresholdFeature thresholdFeature = (BinaryThresholdFeature)feature;

				ContinuousFeature continuousFeature = thresholdFeature.toContinuousFeature();

				// XXX
				encoder.toCategorical(continuousFeature.getName(), null);

				feature = continuousFeature;
			} else

			if(feature instanceof CategoricalFeature){
				CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

				if(hasMissingCategory(featureCategories)){

					if(hasMissingCategory(categoricalFeature.getValues())){
						ClassDictUtil.checkSize(featureCategories, categoricalFeature.getValues());

						featureCategories = new ArrayList<>(categoricalFeature.getValues());
					} else

					{
						ClassDictUtil.checkSize(dropMissingCategory(featureCategories), categoricalFeature.getValues());

						featureCategories = new ArrayList<>(categoricalFeature.getValues());

						DataType dataType = categoricalFeature.getDataType();
						switch(dataType){
							case FLOAT:
							case DOUBLE:
								featureCategories.add(Double.NaN);
								break;
							default:
								featureCategories.add(null);
								break;
						}
					}
				} else

				{
					ClassDictUtil.checkSize(featureCategories, categoricalFeature.getValues());

					featureCategories = new ArrayList<>(categoricalFeature.getValues());
				}
			} else

			if(feature instanceof ObjectFeature){
				ObjectFeature objectFeature = (ObjectFeature)feature;
			} else

			if(feature instanceof WildcardFeature){
				WildcardFeature wildcardFeature = (WildcardFeature)feature;

				if(hasMissingCategory(featureCategories)){
					feature = wildcardFeature.toCategoricalFeature(dropMissingCategory(featureCategories));
				} else

				{
					feature = wildcardFeature.toCategoricalFeature(featureCategories);
				}

				CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

				DataType dataType = TypeUtil.getDataType(categoricalFeature.getValues(), DataType.STRING);

				DataField dataField = (DataField)categoricalFeature.getField();
				if(dataField.requireDataType() != dataType){
					dataField.setDataType(dataType);
				}
			} else

			{
				throw new IllegalArgumentException();
			} // End if

			if(featureInfrequentEnabled){

				if(infrequentCategory == null || featureCategories.contains(infrequentCategory)){
					throw new IllegalArgumentException();
				}

				List<Object> featureInfrequentCategories = selectValues(featureCategories, featureInfrequentIndices);

				featureCategories.removeAll(featureInfrequentCategories);

				feature = EncoderUtil.encodeRegroupFeature(this, feature, featureInfrequentCategories, infrequentCategory, encoder);
			} // End if

			if(dropIdx != null){
				Integer index = dropIdx.get(i);

				if(index != null){
					// Unbox to primitive value in order to ensure correct List#remove(int) vs. List#remove(Object) method resolution
					int intIndex = dropIdx.get(i);

					featureCategories.remove(intIndex);
				}
			}

			for(int j = 0; j < featureCategories.size(); j++){
				Object category = featureCategories.get(j);

				if(EncoderUtil.isMissingCategory(category)){
					result.add(new MissingValueFeature(encoder, feature));
				} else

				{
					result.add(new BinaryFeature(encoder, feature, category));
				}
			}

			if(featureInfrequentEnabled){
				result.add(new BinaryFeature(encoder, feature, infrequentCategory));
			}
		}

		return result;
	}

	public Object getDrop(){
		return getOptionalObject("drop");
	}

	public List<Integer> getDropIdx(){
		List<Number> dropIdx = getNumberArray("drop_idx_");

		if(dropIdx == null){
			return null;
		}

		return Lists.transform(dropIdx, number -> number != null ? ValueUtil.asInteger(number) : null);
	}

	public Boolean getInfrequentEnabled(){
		return getOptionalBoolean("_infrequent_enabled", false);
	}

	public List<List<Number>> getInfrequentIndices(){
		return getArrayList("_infrequent_indices", Number.class);
	}

	static
	private boolean hasMissingCategory(List<?> categories){

		if(!categories.isEmpty()){
			Object lastCategory = categories.get(categories.size() - 1);

			return EncoderUtil.isMissingCategory(lastCategory);
		}

		return false;
	}

	static
	private <E> List<E> dropMissingCategory(List<E> categories){

		if(hasMissingCategory(categories)){
			return categories.subList(0, categories.size() - 1);
		}

		return categories;
	}

	static
	private Object getInfrequentCategory(Feature feature){
		DataType dataType = feature.getDataType();

		switch(dataType){
			case STRING:
				return "infrequent";
			case INTEGER:
			case FLOAT:
			case DOUBLE:
				return -999;
			default:
				return null;
		}
	}

	static
	private <E> List<E> selectValues(List<E> values, Collection<Number> indices){

		if(indices == null || indices.isEmpty()){
			return Collections.emptyList();
		}

		List<E> result = new ArrayList<>();

		for(Number index : indices){
			E value = values.get(ValueUtil.asInt(index));

			result.add(value);
		}

		return result;
	}
}