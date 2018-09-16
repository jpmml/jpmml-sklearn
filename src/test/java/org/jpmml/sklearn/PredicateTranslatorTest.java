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

import java.util.List;

import org.dmg.pmml.CompoundPredicate;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimpleSetPredicate;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class PredicateTranslatorTest extends TranslatorTest {

	@Test
	public void translateLogicalPredicate(){
		CompoundPredicate compoundPredicate = (CompoundPredicate)PredicateTranslator.translate("X[0] > 0.0 and X[1] > 0 or X[2] > 0", doubleFeatures);

		checkCompoundPredicate(compoundPredicate, CompoundPredicate.BooleanOperator.OR, CompoundPredicate.class, SimplePredicate.class);

		compoundPredicate = (CompoundPredicate)PredicateTranslator.translate("(X[\"a\"] > 0.0) and ((X[\"b\"] > 0) or (X[\"c\"] > 0))", doubleFeatures);

		checkCompoundPredicate(compoundPredicate, CompoundPredicate.BooleanOperator.AND, SimplePredicate.class, CompoundPredicate.class);
	}

	@Test
	public void translateComparisonPredicate(){
		SimplePredicate simplePredicate = (SimplePredicate)PredicateTranslator.translate("X['a'] > 0.0", doubleFeatures);

		checkSimplePredicate(simplePredicate, FieldName.create("a"), SimplePredicate.Operator.GREATER_THAN, "0.0");

		try {
			PredicateTranslator.translate("X['a'] > X['b']", doubleFeatures);

			fail();
		} catch(ClassCastException cce){
			// Ignored
		}

		simplePredicate = (SimplePredicate)PredicateTranslator.translate("X[0] is None", doubleFeatures);

		checkSimplePredicate(simplePredicate, FieldName.create("a"), SimplePredicate.Operator.IS_MISSING, null);

		simplePredicate = (SimplePredicate)PredicateTranslator.translate("X['a'] is not None", doubleFeatures);

		checkSimplePredicate(simplePredicate, FieldName.create("a"), SimplePredicate.Operator.IS_NOT_MISSING, null);

		simplePredicate = (SimplePredicate)PredicateTranslator.translate("X[0] == \"one\"", stringFeatures);

		checkSimplePredicate(simplePredicate, FieldName.create("a"), SimplePredicate.Operator.EQUAL, "one");

		SimpleSetPredicate simpleSetPredicate = (SimpleSetPredicate)PredicateTranslator.translate("X[0] in [1, 2, 3]", doubleFeatures);

		checkSimpleSetPredicate(simpleSetPredicate, FieldName.create("a"), SimpleSetPredicate.BooleanOperator.IS_IN, "1 2 3");

		simpleSetPredicate = (SimpleSetPredicate)PredicateTranslator.translate("X[0] in ['one', 'two', 'three']", stringFeatures);

		checkSimpleSetPredicate(simpleSetPredicate, FieldName.create("a"), SimpleSetPredicate.BooleanOperator.IS_IN, "one two three");

		simpleSetPredicate = (SimpleSetPredicate)PredicateTranslator.translate("X['a'] not in [-1.5, -1, -0.5, 0]", doubleFeatures);

		checkSimpleSetPredicate(simpleSetPredicate, FieldName.create("a"), SimpleSetPredicate.BooleanOperator.IS_NOT_IN, "-1.5 -1 -0.5 0");
	}

	static
	private void checkCompoundPredicate(CompoundPredicate compoundPredicate, CompoundPredicate.BooleanOperator booleanOperator, Class<? extends Predicate>... predicateClazzes){
		assertEquals(booleanOperator, compoundPredicate.getBooleanOperator());

		List<Predicate> predicates = compoundPredicate.getPredicates();
		assertEquals(predicateClazzes.length, predicates.size());

		for(int i = 0; i < predicateClazzes.length; i++){
			Class<? extends Predicate> predicateClazz = predicateClazzes[i];
			Predicate predicate = predicates.get(i);

			assertEquals(predicateClazz, predicate.getClass());
		}
	}

	static
	private void checkSimplePredicate(SimplePredicate simplePredicate, FieldName field, SimplePredicate.Operator operator, String value){
		assertEquals(field, simplePredicate.getField());
		assertEquals(operator, simplePredicate.getOperator());
		assertEquals(value, simplePredicate.getValue());
	}

	static
	private void checkSimpleSetPredicate(SimpleSetPredicate simpleSetPredicate, FieldName field, SimpleSetPredicate.BooleanOperator booleanOperator, String value){
		assertEquals(field, simpleSetPredicate.getField());
		assertEquals(booleanOperator, simpleSetPredicate.getBooleanOperator());
		assertEquals(value, (simpleSetPredicate.getArray()).getValue());
	}
}