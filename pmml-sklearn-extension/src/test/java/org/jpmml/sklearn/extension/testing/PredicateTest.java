/*
 * Copyright (c) 2021 Villu Ruusmann
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
package org.jpmml.sklearn.extension.testing;

import java.util.Collections;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimpleSetPredicate;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;
import sklearn.tree.HasTreeOptions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class PredicateTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void checkNumeric() throws Exception {
		Map<String, Object> options = Collections.singletonMap(HasTreeOptions.OPTION_NUMERIC, Boolean.TRUE);

		check("BinaryEncoder", AUDIT, options);
		check("CountEncoder", AUDIT, options);
		check("LeaveOneOutEncoder", AUDIT, options);
		check("TargetEncoder", AUDIT, options);
	}

	@Test
	public void checkNonNumeric() throws Exception {
		Map<String, Object> options = Collections.singletonMap(HasTreeOptions.OPTION_NUMERIC, Boolean.FALSE);

		// XXX
		//check("BinaryEncoder", AUDIT, options);
		//check("CountEncoder", AUDIT, options);
		//check("LeaveOneOutEncoder", AUDIT, options);
		//check("TargetEncoder", AUDIT, options);
	}

	public void check(String name, String dataset, Map<String, Object> options) throws Exception {
		Predicate<ResultField> predicate = (resultField) -> true;
		Equivalence<Object> equivalence = getEquivalence();

		try(SkLearnEncoderBatch batch = (SkLearnEncoderBatch)createBatch(name, dataset, predicate, equivalence)){
			batch.setOptions(options);

			Boolean numeric = (Boolean)options.get(HasTreeOptions.OPTION_NUMERIC);

			check(batch, numeric);
		}
	}

	static
	private void check(SkLearnEncoderBatch batch, boolean numeric) throws Exception {
		PMML pmml = batch.getPMML();

		Visitor visitor = new AbstractVisitor(){

			private int simplePredicateCount = 0;

			private int simpleSetPredicateCount = 0;


			@Override
			public PMMLObject popParent(){
				PMMLObject parent = super.popParent();

				if(parent instanceof PMML){
					PMML pmml = (PMML)parent;

					if(numeric){
						assertTrue(simplePredicateCount > 100);
						assertEquals(0, simpleSetPredicateCount);
					} else

					{
						assertTrue(simplePredicateCount > 100);
						assertTrue(simpleSetPredicateCount > 100);
					}
				}

				return parent;
			}

			@Override
			public VisitorAction visit(SimplePredicate simplePredicate){
				this.simplePredicateCount++;

				return super.visit(simplePredicate);
			}

			@Override
			public VisitorAction visit(SimpleSetPredicate simpleSetPredicate){
				this.simpleSetPredicateCount++;

				return super.visit(simpleSetPredicate);
			}
		};
		visitor.applyTo(pmml);
	}
}