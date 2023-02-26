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
package sklearn2pmml.util;

import java.util.List;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldRef;
import org.jpmml.python.ExpressionTranslator;
import org.jpmml.python.PredicateTranslator;
import org.jpmml.python.Scope;

public class EvaluatableUtil {

	private EvaluatableUtil(){
	}

	static
	public org.dmg.pmml.Expression translateExpression(String expr, List<String> functionDefs, Scope scope){
		ExpressionTranslator expressionTranslator = new ExpressionTranslator(scope);

		for(String functionDef : functionDefs){
			expressionTranslator.addFunctionDef(functionDef);
		}

		if(expr.indexOf('\n') > -1){
			DerivedField derivedField = expressionTranslator.translateDef(expr);

			return new FieldRef(derivedField);
		}

		return expressionTranslator.translateExpression(expr);
	}

	static
	public org.dmg.pmml.Predicate translatePredicate(String expr, List<String> functionDefs, Scope scope){
		PredicateTranslator predicateTranslator = new PredicateTranslator(scope);

		for(String functionDef : functionDefs){
			predicateTranslator.addFunctionDef(functionDef);
		}

		return predicateTranslator.translatePredicate(expr);
	}
}