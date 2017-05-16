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
package sklearn.preprocessing;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.TypeDefinitionField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class ImputerUtil {

	private ImputerUtil(){
	}

	static
	public Feature encodeFeature(Feature feature, Object missingValue, Object replacementValue, MissingValueTreatmentMethod missingValueTreatmentMethod, SkLearnEncoder encoder){
		TypeDefinitionField field = encoder.getField(feature.getName());

		if(field instanceof DataField){
			MissingValueDecorator missingValueDecorator = new MissingValueDecorator()
				.setMissingValueReplacement(ValueUtil.formatValue(replacementValue))
				.setMissingValueTreatment(missingValueTreatmentMethod);

			if(missingValue != null){
				missingValueDecorator.addValues(ValueUtil.formatValue(missingValue));
			}

			encoder.addDecorator(feature.getName(), missingValueDecorator);

			return feature;
		} else

		if(field instanceof DerivedField){
			Expression expression = feature.ref();

			if(missingValue != null){
				expression = PMMLUtil.createApply("equal", expression, PMMLUtil.createConstant(missingValue));
			} else

			{
				expression = PMMLUtil.createApply("isMissing", expression);
			}

			expression = PMMLUtil.createApply("if", expression, PMMLUtil.createConstant(replacementValue), feature.ref());

			DerivedField derivedField = encoder.createDerivedField(FeatureUtil.createName("imputer", feature), expression);

			DataType dataType = derivedField.getDataType();
			switch(dataType){
				case INTEGER:
				case FLOAT:
				case DOUBLE:
					return new ContinuousFeature(encoder, derivedField);
				default:
					return new Feature(encoder, derivedField.getName(), derivedField.getDataType()){

						@Override
						public ContinuousFeature toContinuousFeature(){
							throw new UnsupportedOperationException();
						}
					};
			}
		} else

		{
			throw new IllegalArgumentException();
		}
	}
}