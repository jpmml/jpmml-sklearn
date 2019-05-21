/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.impute;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.sklearn.SkLearnEncoder;

public class ImputerUtil {

	private ImputerUtil(){
	}

	static
	public Feature encodeFeature(Feature feature, Boolean addIndicator, Object missingValue, Object replacementValue, MissingValueTreatmentMethod missingValueTreatmentMethod, SkLearnEncoder encoder){
		Field<?> field = feature.getField();

		if(field instanceof DataField && !addIndicator){
			MissingValueDecorator missingValueDecorator = new MissingValueDecorator()
				.setMissingValueReplacement(replacementValue)
				.setMissingValueTreatment(missingValueTreatmentMethod);

			if(missingValue != null){
				missingValueDecorator.addValues(missingValue);
			}

			encoder.addDecorator(feature.getName(), missingValueDecorator);

			return feature;
		} // End if

		if((field instanceof DataField) || (field instanceof DerivedField)){
			Expression expression = feature.ref();

			if(missingValue != null){
				expression = PMMLUtil.createApply("equal", expression, PMMLUtil.createConstant(missingValue, feature.getDataType()));
			} else

			{
				expression = PMMLUtil.createApply("isMissing", expression);
			}

			expression = PMMLUtil.createApply("if", expression, PMMLUtil.createConstant(replacementValue, feature.getDataType()), feature.ref());

			DerivedField derivedField = encoder.createDerivedField(FeatureUtil.createName("imputer", feature), field.getOpType(), field.getDataType(), expression);

			DataType dataType = derivedField.getDataType();
			switch(dataType){
				case INTEGER:
				case FLOAT:
				case DOUBLE:
					return new ContinuousFeature(encoder, derivedField);
				case STRING:
					return new StringFeature(encoder, derivedField);
				default:
					return new ObjectFeature(encoder, derivedField.getName(), derivedField.getDataType());
			}
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	public Feature encodeIndicatorFeature(Feature feature, Object missingValue, SkLearnEncoder encoder){
		Expression expression = feature.ref();

		if(missingValue != null){
			expression = PMMLUtil.createApply("equal", expression, PMMLUtil.createConstant(missingValue, feature.getDataType()));
		} else

		{
			expression = PMMLUtil.createApply("isMissing", expression);
		}

		DerivedField derivedField = encoder.createDerivedField(FeatureUtil.createName("missing_indicator", feature), OpType.CATEGORICAL, DataType.BOOLEAN, expression);

		return new BooleanFeature(encoder, derivedField);
	}
}
