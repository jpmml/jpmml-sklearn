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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.BinaryThresholdFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueFeature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;

public class MultiOneHotEncoder extends BaseEncoder {

	public MultiOneHotEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<List<?>> categories = getCategories();
		Object drop = getDrop();
		List<Integer> dropIdx = (drop != null ? getDropIdx() : null);
		String handleUnknown = getHandleUnknown();
		Boolean infrequentEnabled = getInfrequentEnabled();
		List<List<Integer>> infrequentIndices = (infrequentEnabled ? getInfrequentIndices() : null);

		ClassDictUtil.checkSize(categories, features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			List<Object> featureCategories = new ArrayList<>(categories.get(i));
			Set<Integer> featureInfrequentIndices = infrequentEnabled ? new LinkedHashSet<>(infrequentIndices.get(i)) : Collections.emptySet();

			InvalidValueTreatmentMethod invalidValueTreatmentMethod;

			switch(handleUnknown){
				case "error":
					invalidValueTreatmentMethod = InvalidValueTreatmentMethod.RETURN_INVALID;
					break;
				case "ignore":
					invalidValueTreatmentMethod = InvalidValueTreatmentMethod.AS_IS;
					break;
				default:
					throw new IllegalArgumentException(handleUnknown);
			}

			EncoderUtil.addDecorator(feature, new InvalidValueDecorator(invalidValueTreatmentMethod, null));

			if(feature instanceof BinaryThresholdFeature){
				BinaryThresholdFeature thresholdFeature = (BinaryThresholdFeature)feature;

				ContinuousFeature continuousFeature = thresholdFeature.toContinuousFeature();

				// XXX
				encoder.toCategorical(continuousFeature.getName(), null);

				feature = continuousFeature;
			} else

			if(feature instanceof CategoricalFeature){
				CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

				if(hasNaNCategory(featureCategories)){
					ClassDictUtil.checkSize(dropNaNCategory(featureCategories), categoricalFeature.getValues());

					featureCategories = new ArrayList<>(categoricalFeature.getValues());
					featureCategories.add(Double.NaN);
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

				if(hasNaNCategory(featureCategories)){
					feature = wildcardFeature.toCategoricalFeature(dropNaNCategory(featureCategories));
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
			}

			Object infrequentCategory = null;

			if(infrequentEnabled){
				infrequentCategory = getInfrequentCategory(feature);

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
				Object featureCategory = featureCategories.get(j);

				if(ValueUtil.isNaN(featureCategory)){
					result.add(new MissingValueFeature(encoder, feature));
				} else

				{
					result.add(new BinaryFeature(encoder, feature, featureCategory));
				}
			}

			if(infrequentEnabled){
				result.add(new BinaryFeature(encoder, feature, infrequentCategory));
			}
		}

		return result;
	}

	public Object getDrop(){
		return getOptionalObject("drop");
	}

	public List<Integer> getDropIdx(){
		List<? extends Number> dropIdx = getNumberArray("drop_idx_");

		if(dropIdx == null){
			return null;
		}

		return Lists.transform(dropIdx, number -> number != null ? ValueUtil.asInteger(number) : null);
	}

	public Boolean getInfrequentEnabled(){
		return getOptionalBoolean("_infrequent_enabled", false);
	}

	public List<List<Integer>> getInfrequentIndices(){
		return EncoderUtil.transformInfrequentIndices(getList("_infrequent_indices", HasArray.class));
	}

	static
	private boolean hasNaNCategory(List<?> categories){

		if(!categories.isEmpty()){
			Object lastCategory = categories.get(categories.size() - 1);

			return ValueUtil.isNaN(lastCategory);
		}

		return false;
	}

	static
	private <E> List<E> dropNaNCategory(List<E> categories){

		if(hasNaNCategory(categories)){
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
	private <E> List<E> selectValues(List<E> values, Collection<Integer> indices){
		List<E> result = new ArrayList<>();

		for(Integer index : indices){
			E value = values.get(index);

			result.add(value);
		}

		return result;
	}
}