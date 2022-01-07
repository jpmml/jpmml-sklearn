/*
 * Copyright (c) 2020 Villu Ruusmann
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

import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.model.visitors.AbstractVisitor;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class FeatureImportancesTest extends MarkupTest implements SkLearnAlgorithms, SkLearnDatasets {

	@Test
	public void checkDecisionTreeAudit() throws Exception {
		check(DECISION_TREE, AUDIT);
	}

	@Test
	public void checkExtraTreesAudit() throws Exception {
		check(EXTRA_TREES, AUDIT);
	}

	@Test
	public void checkGradientBoostingAudit() throws Exception {
		check(GRADIENT_BOOSTING, AUDIT);
	}

	@Test
	public void checkRandomForestAudit() throws Exception {
		check(RANDOM_FOREST, AUDIT);
	}

	@Test
	public void checkXGBoostAudit() throws Exception {
		check(XGB, AUDIT);
	}

	@Override
	public void check(PMML pmml){
		Visitor visitor = new AbstractVisitor(){

			@Override
			public VisitorAction visit(MiningModel miningModel){
				PMMLObject parent = getParent();

				if(parent instanceof PMML){
					String algorithmName = miningModel.getAlgorithmName();

					if(algorithmName != null && algorithmName.contains("XGBoost")){
						check(miningModel, null);

						return VisitorAction.TERMINATE;
					} else

					{
						check(miningModel, 1d);
					}
				}

				return super.visit(miningModel);
			}

			@Override
			public VisitorAction visit(TreeModel treeModel){
				check(treeModel, 1d);

				return super.visit(treeModel);
			}

			private void check(Model model, Number expectedSum){
				double sum = 0d;

				MiningSchema miningSchema = model.requireMiningSchema();
				if(miningSchema.hasMiningFields()){
					List<MiningField> miningFields = miningSchema.getMiningFields();

					for(MiningField miningField : miningFields){
						Number importance = miningField.getImportance();
						MiningField.UsageType usageType = miningField.getUsageType();

						switch(usageType){
							case ACTIVE:
								{
									assertNotNull(importance);

									sum += importance.doubleValue();
								}
								break;
							default:
								break;
						}
					}
				}

				if(expectedSum != null){
					assertEquals(expectedSum.doubleValue(), sum, 1e-13);
				}
			}
		};
		visitor.applyTo(pmml);
	}
}