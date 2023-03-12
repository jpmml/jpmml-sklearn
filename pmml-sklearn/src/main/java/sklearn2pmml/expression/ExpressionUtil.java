/*
 * Copyright (c) 2023 Villu Ruusmann
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
package sklearn2pmml.expression;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.PMMLEncoder;

public class ExpressionUtil {

	private ExpressionUtil(){
	}

	static
	public ContinuousFeature toFeature(String name, Expression expression, PMMLEncoder encoder){
		DerivedField derivedField = null;

		if(expression instanceof FieldRef){
			FieldRef fieldRef = (FieldRef)expression;

			derivedField = encoder.getDerivedField(fieldRef.requireField());

			if(derivedField != null){
				DataType dataType = derivedField.requireDataType();
				OpType opType = derivedField.requireOpType();

				switch(dataType){
					case INTEGER:
					case FLOAT:
					case DOUBLE:
						break;
					default:
						derivedField = null;
						break;
				} // End switch

				switch(opType){
					case CONTINUOUS:
						break;
					default:
						derivedField = null;
						break;
				}
			}
		} // End if

		if(derivedField == null){
			derivedField = encoder.createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
		}

		return new ContinuousFeature(encoder, derivedField);
	}
}