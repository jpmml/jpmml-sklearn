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
package org.jpmml.sklearn;

import java.util.Deque;
import java.util.Iterator;

import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.visitors.AbstractVisitor;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class ModelStatsTest extends MarkupTest implements Algorithms, SkLearnDatasets {

	@Test
	public void checkGBDTLRAudit() throws Exception {
		check(GBDT_LR, AUDIT);
	}

	@Test
	public void checkTPOTAuto() throws Exception {
		check("TPOT", AUTO);
	}

	@Test
	public void checkIsolationForestHousing() throws Exception {
		check(ISOLATION_FOREST, HOUSING);
	}

	@Test
	public void checkEstimatorTransformerWheat() throws Exception {
		check("EstimatorTransformer", WHEAT);
	}

	@Override
	public void check(PMML pmml){
		Visitor visitor = new AbstractVisitor(){

			@Override
			public VisitorAction visit(ModelStats modelStats){
				Deque<PMMLObject> parents = getParents();

				Iterator<PMMLObject> it = parents.iterator();

				assertTrue(it.next() instanceof Model);
				assertTrue(it.next() instanceof PMML);

				return super.visit(modelStats);
			}
		};
		visitor.applyTo(pmml);
	}
}