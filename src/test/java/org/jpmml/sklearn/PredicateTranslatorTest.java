/*
 * Copyright (c) 2018 Villu Ruusmann
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

import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.Array;
import org.dmg.pmml.ComplexArray;
import org.dmg.pmml.CompoundPredicate;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimpleSetPredicate;
import org.jpmml.converter.Feature;
import org.jpmml.model.ReflectionUtil;
import org.junit.Test;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class PredicateTranslatorTest extends TranslatorTest {

	@Test
	public void translateLogicalPredicate(){
		String string = "X[0] > 0.0 and X[1] > 0 or X[2] > 0";

		Predicate first = new SimplePredicate(FieldName.create("a"), SimplePredicate.Operator.GREATER_THAN, "0.0");
		Predicate second = new SimplePredicate(FieldName.create("b"), SimplePredicate.Operator.GREATER_THAN, "0");
		Predicate third = new SimplePredicate(FieldName.create("c"), SimplePredicate.Operator.GREATER_THAN, "0");

		Predicate expected = new CompoundPredicate(CompoundPredicate.BooleanOperator.OR, null)
			.addPredicates(new CompoundPredicate(CompoundPredicate.BooleanOperator.AND, null)
				.addPredicates(first)
				.addPredicates(second)
			)
			.addPredicates(third);

		checkPredicate(expected, string, doubleFeatures);

		string = "(X[\"a\"] > 0.0) and ((X[\"b\"] > 0) or (X[\"c\"] > 0))";

		expected = new CompoundPredicate(CompoundPredicate.BooleanOperator.AND, null)
			.addPredicates(first)
			.addPredicates(new CompoundPredicate(CompoundPredicate.BooleanOperator.OR, null)
				.addPredicates(second, third)
			);

		checkPredicate(expected, string, doubleFeatures);
	}

	@Test
	public void translateComparisonPredicate(){
		Predicate expected = new SimplePredicate(FieldName.create("a"), SimplePredicate.Operator.GREATER_THAN, "0.0");

		checkPredicate(expected, "X['a'] > 0.0", doubleFeatures);

		try {
			PredicateTranslator.translate("X['a'] > X['b']", doubleFeatures);

			fail();
		} catch(ClassCastException cce){
			// Ignored
		}

		expected = new SimplePredicate(FieldName.create("a"), SimplePredicate.Operator.IS_MISSING, null);
		checkPredicate(expected, "X[0] is None", doubleFeatures);

		expected = new SimplePredicate(FieldName.create("a"), SimplePredicate.Operator.IS_NOT_MISSING, null);
		checkPredicate(expected, "X[0] is not None", doubleFeatures);

		expected = new SimplePredicate(FieldName.create("a"), SimplePredicate.Operator.EQUAL, "one");
		checkPredicate(expected, "X[0] == \"one\"", stringFeatures);

		expected = createSimpleSetPredicate(FieldName.create("a"), SimpleSetPredicate.BooleanOperator.IS_IN, Arrays.asList("1", "2", "3"));
		checkPredicate(expected, "X[0] in [1, 2, 3]", doubleFeatures);

		expected = createSimpleSetPredicate(FieldName.create("a"), SimpleSetPredicate.BooleanOperator.IS_IN, Arrays.asList("one", "two", "three"));
		checkPredicate(expected, "X[0] in ['one', 'two', 'three']", stringFeatures);

		expected = createSimpleSetPredicate(FieldName.create("a"), SimpleSetPredicate.BooleanOperator.IS_NOT_IN, Arrays.asList("-1.5", "-1", "-0.5", "0"));
		checkPredicate(expected, "X['a'] not in [-1.5, -1, -0.5, 0]", doubleFeatures);
	}

	static
	private void checkPredicate(Predicate expected, String string, List<Feature> features){
		Predicate actual = PredicateTranslator.translate(string, features);

		assertTrue(ReflectionUtil.equals(expected, actual));
	}

	static
	private SimpleSetPredicate createSimpleSetPredicate(FieldName field, SimpleSetPredicate.BooleanOperator booleanOperator, List<String> values){
		Array array = new ComplexArray()
			.setType(Array.Type.STRING)
			.setValue(values);

		return new SimpleSetPredicate(field, booleanOperator, array);
	}
}