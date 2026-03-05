/*
 * Copyright (c) 2026 Villu Ruusmann
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
package causalml.meta;

import java.util.Arrays;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Target;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.dmg.pmml.tree.Node;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.visitors.AbstractVisitor;

public class BaseSRegressor extends BaseSLearner {

	public BaseSRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeBinaryModel(Model treatmentModel, Model controlModel, Schema schema){
		Visitor scoreNegater = new AbstractVisitor(){

			@Override
			public VisitorAction visit(Node node){
				Number score = (Number)node.requireScore();

				if(score.doubleValue() != 0d){
					node.setScore(ValueUtil.toNegative(score));
				}

				return super.visit(node);
			}

			@Override
			public VisitorAction visit(Target target){
				Number rescaleConstant = target.getRescaleConstant();

				if(rescaleConstant != null && rescaleConstant.doubleValue() != 0d){
					target.setRescaleConstant((Number)ValueUtil.toNegative(rescaleConstant));
				}

				return super.visit(target);
			}
		};
		scoreNegater.applyTo(controlModel);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.SUM, Segmentation.MissingPredictionTreatment.RETURN_MISSING, Arrays.asList(treatmentModel, controlModel)));

		return miningModel;
	}
}