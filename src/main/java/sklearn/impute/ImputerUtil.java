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

import java.util.Collections;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.Value;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class ImputerUtil {

	private ImputerUtil(){
	}

	static
	public Feature encodeFeature(Transformer transformer, Feature feature, Boolean addIndicator, Object missingValue, Object replacementValue, MissingValueTreatmentMethod missingValueTreatmentMethod, SkLearnEncoder encoder){
		Field<?> field = feature.getField();

		if(field instanceof DataField && !addIndicator){
			DataField dataField = (DataField)field;

			encoder.addDecorator(dataField, new MissingValueDecorator(missingValueTreatmentMethod, replacementValue));

			if(missingValue != null){
				PMMLUtil.addValues(dataField, Collections.singletonList(missingValue), Value.Property.MISSING);
			}

			return feature;
		} // End if

		if((field instanceof DataField) || (field instanceof DerivedField)){
			Expression expression = feature.ref();

			if(missingValue != null){
				expression = PMMLUtil.createApply(PMMLFunctions.EQUAL, expression, PMMLUtil.createConstant(missingValue, feature.getDataType()));
			} else

			{
				expression = PMMLUtil.createApply(PMMLFunctions.ISMISSING, expression);
			}

			expression = PMMLUtil.createApply(PMMLFunctions.IF)
				.addExpressions(expression)
				.addExpressions(PMMLUtil.createConstant(replacementValue, feature.getDataType()), feature.ref());

			DerivedField derivedField = encoder.createDerivedField(transformer.createFieldName("imputer", feature), field.getOpType(), field.getDataType(), expression);

			DataType dataType = derivedField.getDataType();
			switch(dataType){
				case INTEGER:
				case FLOAT:
				case DOUBLE:
					return new ContinuousFeature(encoder, derivedField);
				case STRING:
					return new StringFeature(encoder, derivedField);
				default:
					return new ObjectFeature(encoder, derivedField);
			}
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	public Feature encodeIndicatorFeature(Transformer transformer, Feature feature, Object missingValue, SkLearnEncoder encoder){
		Expression expression = feature.ref();

		if(missingValue != null){
			expression = PMMLUtil.createApply(PMMLFunctions.EQUAL, expression, PMMLUtil.createConstant(missingValue, feature.getDataType()));
		} else

		{
			expression = PMMLUtil.createApply(PMMLFunctions.ISMISSING, expression);
		}

		DerivedField derivedField = encoder.createDerivedField(transformer.createFieldName("missingIndicator", feature), OpType.CATEGORICAL, DataType.BOOLEAN, expression);

		return new BooleanFeature(encoder, derivedField);
	}
}
