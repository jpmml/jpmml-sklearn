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
package org.jpmml.sklearn;

import java.util.List;

import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.model.ReflectionUtil;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class ExpressionTranslatorTest extends TranslatorTest {

	@Test
	public void translateLogicalExpression(){
		String string = "X[\"a\"] and X[\"b\"] or X[\"c\"]";

		Expression expected = PMMLUtil.createApply(PMMLFunctions.OR)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.AND)
				.addExpressions(new FieldRef(FieldName.create("a")), new FieldRef(FieldName.create("b")))
			)
			.addExpressions(new FieldRef(FieldName.create("c")));

		checkExpression(expected, string, booleanFeatures);

		string = "not X[\"a\"]";

		expected = PMMLUtil.createApply(PMMLFunctions.NOT)
			.addExpressions(new FieldRef(FieldName.create("a")));

		checkExpression(expected, string, booleanFeatures);
	}

	@Test
	public void translateIdentityComparisonExpression(){
		String string = "X[0] is None";

		FieldRef first = new FieldRef(FieldName.create("a"));

		Expression expected = PMMLUtil.createApply(PMMLFunctions.ISMISSING)
			.addExpressions(first);

		checkExpression(expected, string, doubleFeatures);

		string = "(X['a'] + 1) is not None";

		expected = PMMLUtil.createApply(PMMLFunctions.ISNOTMISSING)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.ADD)
				.addExpressions(first, PMMLUtil.createConstant("1", DataType.INTEGER))
			);

		checkExpression(expected, string, doubleFeatures);
	}

	@Test
	public void translateComparisonExpression(){
		String string = "X['a'] and X['b']";

		FieldRef first = new FieldRef(FieldName.create("a"));
		FieldRef second = new FieldRef(FieldName.create("b"));

		Expression expected = PMMLUtil.createApply(PMMLFunctions.AND)
			.addExpressions(first, second);

		checkExpression(expected, string, booleanFeatures);

		string = "X['a'] == True and X['b'] == False";

		expected = PMMLUtil.createApply(PMMLFunctions.AND)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.EQUAL)
				.addExpressions(first, PMMLUtil.createConstant("true", DataType.BOOLEAN))
			)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.EQUAL)
				.addExpressions(second, PMMLUtil.createConstant("false", DataType.BOOLEAN))
			);

		checkExpression(expected, string, booleanFeatures);

		string = "X[0] in [0.0, 1.0]";

		expected = PMMLUtil.createApply(PMMLFunctions.ISIN)
			.addExpressions(first, PMMLUtil.createConstant("0.0", DataType.DOUBLE), PMMLUtil.createConstant("1.0", DataType.DOUBLE));

		checkExpression(expected, string, doubleFeatures);

		string = "(X[0] + 1.0) not in [X[1]]";

		expected = PMMLUtil.createApply(PMMLFunctions.ISNOTIN)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.ADD)
				.addExpressions(first, PMMLUtil.createConstant("1.0", DataType.DOUBLE))
			)
			.addExpressions(second);

		checkExpression(expected, string, doubleFeatures);

		string = "X[\"a\"] > X[\"b\"]";

		expected = PMMLUtil.createApply(PMMLFunctions.GREATERTHAN)
			.addExpressions(first, second);

		checkExpression(expected, string, doubleFeatures);

		string = "not X[\"a\"] < 0.0";

		expected = PMMLUtil.createApply(PMMLFunctions.NOT)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.LESSTHAN)
				.addExpressions(first, PMMLUtil.createConstant("0.0", DataType.DOUBLE))
			);

		checkExpression(expected, string, doubleFeatures);
	}

	@Test
	public void translateArithmeticExpression(){
		String string = "(X[0] + X[1] - 1.0) / X[2] * -2";

		Expression expected = PMMLUtil.createApply(PMMLFunctions.MULTIPLY)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.DIVIDE)
				.addExpressions(PMMLUtil.createApply(PMMLFunctions.SUBTRACT)
					.addExpressions(PMMLUtil.createApply(PMMLFunctions.ADD)
						.addExpressions(new FieldRef(FieldName.create("a")), new FieldRef(FieldName.create("b")))
					)
					.addExpressions(PMMLUtil.createConstant("1.0", DataType.DOUBLE))
				)
				.addExpressions(new FieldRef(FieldName.create("c")))
			)
			.addExpressions(PMMLUtil.createConstant("-2", DataType.INTEGER));

		checkExpression(expected, string, doubleFeatures);

		string = "(X[\"a\"] + X[\"b\"] - 1.0) / X['c'] * -2";

		checkExpression(expected, string, doubleFeatures);
	}

	@Test
	public void translateStringConcatenationExpression(){
		String string = "\'19\' + X[0] + \'-01-01\'";

		Expression expected = PMMLUtil.createApply(PMMLFunctions.CONCAT)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.CONCAT)
				.addExpressions(PMMLUtil.createConstant("19", DataType.STRING), new FieldRef(FieldName.create("a")))
			)
			.addExpressions(PMMLUtil.createConstant("-01-01", DataType.STRING));

		checkExpression(expected, string, stringFeatures);

		string = "\"19\" + X[\'a\'] + \"-01-01\"";

		checkExpression(expected, string, stringFeatures);
	}

	@Test
	public void translateUnaryExpression(){
		Constant minusOne = PMMLUtil.createConstant("-1", DataType.INTEGER);
		Constant plusOne = PMMLUtil.createConstant("1", DataType.INTEGER);

		checkExpression(minusOne, "-1", doubleFeatures);

		checkExpression(plusOne, "1", doubleFeatures);
		checkExpression(plusOne, "+1", doubleFeatures);

		checkExpression(minusOne, "-+1", doubleFeatures);
		checkExpression(plusOne, "--1", doubleFeatures);
		checkExpression(minusOne, "---1", doubleFeatures);
	}

	@Test
	public void translateFunctionInvocationExpression(){
		String string = "X[\"a\"] if pandas.notnull(X[\"a\"]) else X[\"b\"] + X[\"c\"]";

		Expression expected = PMMLUtil.createApply(PMMLFunctions.IF)
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.ISNOTMISSING)
				.addExpressions(new FieldRef(FieldName.create("a")))
			)
			.addExpressions(new FieldRef(FieldName.create("a")))
			.addExpressions(PMMLUtil.createApply(PMMLFunctions.ADD)
				.addExpressions(new FieldRef(FieldName.create("b")), new FieldRef(FieldName.create("c")))
			);

		checkExpression(expected, string, doubleFeatures);
	}

	static
	private void checkExpression(Expression expected, String string, List<Feature> features){
		Expression actual = ExpressionTranslator.translate(string, features, false);

		assertTrue(ReflectionUtil.equals(expected, actual));
	}
}