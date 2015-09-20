/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.tree;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.FieldTypeAnalyzer;

public class TreeFieldTypeAnalyzer extends FieldTypeAnalyzer {

	@Override
	public VisitorAction visit(SimplePredicate simplePredicate){
		FieldName field = simplePredicate.getField();
		SimplePredicate.Operator operator = simplePredicate.getOperator();
		String value = simplePredicate.getValue();

		if((SimplePredicate.Operator.LESS_OR_EQUAL).equals(operator) && ("0.5").equals(value)){
			addDataType(field, DataType.BOOLEAN);
		} else

		if((SimplePredicate.Operator.GREATER_THAN).equals(operator) && ("0.5").equals(value)){
			addDataType(field, DataType.BOOLEAN);
		} else

		{
			addDataType(field, DataType.FLOAT);
		}

		return super.visit(simplePredicate);
	}
}