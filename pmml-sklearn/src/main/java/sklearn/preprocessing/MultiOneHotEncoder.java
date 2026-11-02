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
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Error;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.BinaryThresholdFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Decorator;
import org.jpmml.converter.DiscreteFeature;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.MissingValueFeature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.UnsupportedFeatureException;
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

			Decorator invalidValueDecorator;

			switch(handleUnknown){
				case MultiOneHotEncoder.HANDLEUNKNOWN_ERROR:
					{
						invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.RETURN_INVALID, null);
					}
					break;
				case MultiOneHotEncoder.HANDLEUNKNOWN_IGNORE:
				case MultiOneHotEncoder.HANDLEUNKNOWN_INFREQUENT_IF_EXIST:
				case MultiOneHotEncoder.HANDLEUNKNOWN_WARN:
					{
						invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_IS, null);
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

			if(feature instanceof DiscreteFeature){
				DiscreteFeature discreteFeature = (DiscreteFeature)feature;

				if(hasMissingCategory(featureCategories)){

					if(hasMissingCategory(discreteFeature.getValues())){
						discreteFeature.expectCardinality(featureCategories.size());

						featureCategories = new ArrayList<>(discreteFeature.getValues());
					} else

					{
						discreteFeature.expectCardinality(dropMissingCategory(featureCategories).size());

						featureCategories = new ArrayList<>(discreteFeature.getValues());

						DataType dataType = discreteFeature.getDataType();
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
					discreteFeature.expectCardinality(featureCategories.size());

					featureCategories = new ArrayList<>(discreteFeature.getValues());
				}
			} else

			if(feature instanceof ObjectFeature){
				ObjectFeature objectFeature = (ObjectFeature)feature;
			} else

			if(feature instanceof WildcardFeature){
				WildcardFeature wildcardFeature = (WildcardFeature)feature;

				List<Object> values = dropMissingCategory(featureCategories);

				feature = wildcardFeature.toCategoricalFeature(values);

				CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

				DataType dataType = TypeUtil.getDataType(categoricalFeature.getValues(), DataType.STRING);

				DataField dataField = (DataField)categoricalFeature.getField();
				if(dataField.requireDataType() != dataType){
					dataField.setDataType(dataType);
				}
			} else

			{
				throw new UnsupportedFeatureException("Expected a categorical or categorical-like feature, got " + ExceptionUtil.formatFeature(feature));
			}

			List<Object> frequentCategories = new ArrayList<>(featureCategories);
			List<Object> infrequentCategories = Collections.emptyList();

			if(featureInfrequentEnabled){
				infrequentCategories = selectValues(featureCategories, featureInfrequentIndices);

				frequentCategories.removeAll(infrequentCategories);
			}

			List<Object> binaryCategories = new ArrayList<>(frequentCategories);

			boolean dropInfrequent = false;

			if(dropIdx != null){
				Integer index = dropIdx.get(i);

				if(index != null){
					Object dropCategory = featureCategories.get(index);

					if(infrequentCategories.contains(dropCategory)){
						dropInfrequent = true;
					} else

					{
						binaryCategories.remove(dropCategory);
					}
				}
			}

			switch(handleUnknown){
				case MultiOneHotEncoder.HANDLEUNKNOWN_ERROR:
					{
						// Scikit-Learn treats any value that was not present in the training dataset as unknown.
						// Therefore, a one-hot encoder that was trained on a dense dataset treats missing values the same way as invalid values.
						feature = toValid(feature, featureCategories, encoder);
					}
					break;
				default:
					break;
			}

			for(int j = 0; j < binaryCategories.size(); j++){
				Object category = binaryCategories.get(j);

				if(EncoderUtil.isMissingCategory(category)){
					result.add(new MissingValueFeature(encoder, feature));
				} else

				{
					result.add(new BinaryFeature(encoder, feature, category));
				}
			}

			if(featureInfrequentEnabled){

				if(!dropInfrequent){
					Feature infrequentFeature = encodeInfrequentFeature(feature, featureCategories, frequentCategories, infrequentCategories, encoder);

					result.add(infrequentFeature);
				}
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

	public String getHandleUnknown(){
		return getEnum("handle_unknown", this::getString, Arrays.asList(MultiOneHotEncoder.HANDLEUNKNOWN_ERROR, MultiOneHotEncoder.HANDLEUNKNOWN_IGNORE, MultiOneHotEncoder.HANDLEUNKNOWN_INFREQUENT_IF_EXIST, MultiOneHotEncoder.HANDLEUNKNOWN_WARN));
	}

	public Boolean getInfrequentEnabled(){
		return getOptionalBoolean("_infrequent_enabled", false);
	}

	public List<List<Number>> getInfrequentIndices(){
		return getArrayList("_infrequent_indices", Number.class);
	}

	private Feature toValid(Feature feature, List<Object> categories, SkLearnEncoder encoder){

		if(feature instanceof DiscreteFeature){
			DiscreteFeature discreteFeature = (DiscreteFeature)feature;

			Feature valueFeature;

			try {
				valueFeature = discreteFeature.toCategoricalFeature();
			} catch(UnsupportedOperationException uoe){
				valueFeature = feature;
			}

			// XXX
			if(valueFeature != feature){
				return feature;
			}
		}

		Field<?> field = feature.getField();

		if(field instanceof DataField){

			if(!hasMissingCategory(categories)){
				encoder.addDecorator(field, new MissingValueDecorator(MissingValueTreatmentMethod.RETURN_INVALID, null));
			}

			return feature;
		} else

		{
			return encodeValidFeature(feature, categories, encoder);
		}
	}

	private Feature encodeValidFeature(Feature feature, List<Object> categories, SkLearnEncoder encoder){
		DataType dataType = feature.getDataType();

		List<Object> validCategories = EncoderUtil.filterCategories(categories);

		Apply validApply = ExpressionUtil.createApply(PMMLFunctions.IF,
			ExpressionUtil.createValueApply(feature.ref(), dataType, validCategories),
			feature.ref(),
			createError("Field \"" + feature.getName() + "\" cannot accept invalid value")
		);

		Expression missingExpression;

		if(hasMissingCategory(categories)){
			missingExpression = feature.ref();
		} else

		{
			missingExpression = createError("Field \"" + feature.getName() + "\" cannot accept missing value");
		}

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.IF,
			ExpressionUtil.createApply(PMMLFunctions.ISMISSING, feature.ref()),
			missingExpression,
			validApply
		);

		DerivedField derivedField = encoder.createDerivedField(createFieldName("valid", feature), OpType.CATEGORICAL, dataType, apply);

		if(feature instanceof DiscreteFeature){
			DiscreteFeature discreteFeature = (DiscreteFeature)feature;

			List<?> values = discreteFeature.getValues();

			return new CategoricalFeature(encoder, derivedField, values);
		} else

		{
			return FeatureUtil.createFeature(derivedField, encoder);
		}
	}

	private Feature encodeInfrequentFeature(Feature feature, List<Object> categories, List<Object> frequentCategories, List<Object> infrequentCategories, SkLearnEncoder encoder){
		String handleUnknown = getHandleUnknown();

		boolean unknownInfrequent;

		switch(handleUnknown){
			case MultiOneHotEncoder.HANDLEUNKNOWN_ERROR:
			case MultiOneHotEncoder.HANDLEUNKNOWN_IGNORE:
				unknownInfrequent = false;
				break;
			case MultiOneHotEncoder.HANDLEUNKNOWN_INFREQUENT_IF_EXIST:
			case MultiOneHotEncoder.HANDLEUNKNOWN_WARN:
				unknownInfrequent = true;
				break;
			default:
				throw new IllegalArgumentException(handleUnknown);
		}

		boolean missingInfrequent;

		if(hasMissingCategory(categories)){
			missingInfrequent = hasMissingCategory(infrequentCategories);
		} else

		{
			missingInfrequent = unknownInfrequent;
		}

		DataType dataType = feature.getDataType();

		Expression valueExpression;

		if(unknownInfrequent){
			List<Object> values = EncoderUtil.filterCategories(frequentCategories);

			if(values.isEmpty()){
				valueExpression = createIndicatorConstant(true);
			} else

			{
				Apply apply = ExpressionUtil.createApply(PMMLFunctions.ISNOTIN, feature.ref());

				for(int i = 0; i < values.size(); i++){
					Object value = values.get(i);

					Constant constant = ExpressionUtil.createConstant(dataType, value);

					apply.addExpressions(constant);
				}

				valueExpression = createIndicator(apply);
			}
		} else

		{
			List<Object> values = EncoderUtil.filterCategories(infrequentCategories);

			if(values.isEmpty()){
				valueExpression = createIndicatorConstant(false);
			} else

			{
				Apply apply = ExpressionUtil.createValueApply(feature.ref(), dataType, values);

				valueExpression = createIndicator(apply);
			}
		}

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.IF,
			ExpressionUtil.createApply(PMMLFunctions.ISMISSING, feature.ref()),
			createIndicatorConstant(missingInfrequent),
			valueExpression
		);

		DerivedField derivedField = encoder.createDerivedField(createFieldName("infrequent", feature), OpType.CATEGORICAL, DataType.INTEGER, apply);

		CategoricalFeature categoricalFeature = new CategoricalFeature(encoder, derivedField, Arrays.asList(0, 1));

		return new BinaryFeature(encoder, categoricalFeature, 1);
	}

	static
	private Error createError(String message){
		Error error = new Error()
			.setMessage(message);

		return error;
	}

	static
	private Expression createIndicator(Expression expression){
		return ExpressionUtil.createApply(PMMLFunctions.IF,
			expression,
			createIndicatorConstant(true),
			createIndicatorConstant(false)
		);
	}

	static
	private Expression createIndicatorConstant(boolean value){
		return ExpressionUtil.createConstant(DataType.INTEGER, value ? 1 : 0);
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
	private <E> List<E> selectValues(List<E> values, Collection<Number> indices){
		List<E> result = new ArrayList<>();

		for(Number index : indices){
			E value = values.get(ValueUtil.asInt(index));

			result.add(value);
		}

		return result;
	}

	private static final String HANDLEUNKNOWN_ERROR = "error";
	private static final String HANDLEUNKNOWN_IGNORE = "ignore";
	private static final String HANDLEUNKNOWN_INFREQUENT_IF_EXIST = "infrequent_if_exist";
	private static final String HANDLEUNKNOWN_WARN = "warn";
}