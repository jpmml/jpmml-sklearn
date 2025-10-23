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

import java.util.Collections;
import java.util.List;

import org.jpmml.python.AbstractTranslator;
import org.jpmml.python.ExpressionTranslator;
import org.jpmml.python.PredicateTranslator;
import org.jpmml.python.Scope;

public class EvaluatableUtil {

	private EvaluatableUtil(){
	}

	static
	public String toString(Object expr){

		if(expr instanceof Evaluatable){
			Evaluatable<?> evaluatable = (Evaluatable<?>)expr;

			return evaluatable.getExpr();
		}

		return (String)expr;
	}

	static
	public org.dmg.pmml.Expression translateExpression(Object expr, Scope scope){

		if(expr instanceof Expression){
			Expression expression = (Expression)expr;

			return expression.translate(scope);
		}

		return translateExpression((String)expr, Collections.emptyList(), scope);
	}

	static
	public org.dmg.pmml.Expression translateExpression(String expr, List<String> functionDefs, Scope scope){
		ExpressionTranslator expressionTranslator = new ExpressionTranslator(scope);
		init(expressionTranslator, functionDefs);

		return expressionTranslator.translateExpression(expr);
	}

	static
	public org.dmg.pmml.Predicate translatePredicate(Object expr, Scope scope){

		if(expr instanceof Predicate){
			Predicate predicate = (Predicate)expr;

			return predicate.translate(scope);
		}

		return translatePredicate((String)expr, Collections.emptyList(), scope);
	}

	static
	public org.dmg.pmml.Predicate translatePredicate(String expr, List<String> functionDefs, Scope scope){
		PredicateTranslator predicateTranslator = new PredicateTranslator(scope);
		init(predicateTranslator, functionDefs);

		return predicateTranslator.translatePredicate(expr);
	}

	static
	private <T extends AbstractTranslator> void init(T translator, List<String> functionDefs){
		translator.registerModuleAliases();

		for(String functionDef : functionDefs){
			translator.addFunctionDef(functionDef);
		}
	}
}