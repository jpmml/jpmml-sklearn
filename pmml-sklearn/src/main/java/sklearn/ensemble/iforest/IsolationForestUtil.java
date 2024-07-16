/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sklearn.ensemble.iforest;

import java.util.Deque;
import java.util.List;

import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.transformations.AbstractTransformation;
import org.jpmml.model.visitors.AbstractVisitor;
import sklearn.Estimator;
import sklearn.OutlierDetector;
import sklearn.OutlierDetectorUtil;
import sklearn.tree.Tree;

public class IsolationForestUtil {

	private IsolationForestUtil(){
	}

	static
	public <E extends Estimator & HasIsolationForest & OutlierDetector> MiningModel encodeMiningModel(E estimator, List<TreeModel> treeModels, boolean corrected, boolean nodeSampleCorrected, Schema schema){
		// "rawAnomalyScore / averagePathLength(maxSamples)"
		Transformation normalizedAnomalyScore = new AbstractTransformation(){

			@Override
			public String getName(String name){
				return "normalizedAnomalyScore";
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				double maxSamples = estimator.getMaxSamples();

				double averagePathLength = (corrected ? correctedAveragePathLength(maxSamples, nodeSampleCorrected) : averagePathLength(maxSamples));

				return ExpressionUtil.createApply(PMMLFunctions.DIVIDE, fieldRef, ExpressionUtil.createConstant(averagePathLength));
			}
		};

		// "0.5 - 2 ^ (-1 * normalizedAnomalyScore)"
		Transformation decisionFunction = new AbstractTransformation(){

			@Override
			public String getName(String name){
				return estimator.getDecisionFunctionField();
			}

			@Override
			public boolean isFinalResult(){
				return true;
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				Number offset = estimator.getOffset();

				return ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, ExpressionUtil.createConstant(-offset.doubleValue()), ExpressionUtil.createApply(PMMLFunctions.POW, ExpressionUtil.createConstant(2d), ExpressionUtil.toNegative(fieldRef)));
			}
		};

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.AVERAGE, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels))
			.setOutput(OutlierDetectorUtil.createPredictedOutput(estimator, "rawAnomalyScore", normalizedAnomalyScore, decisionFunction));

		return miningModel;
	}

	static
	public void transformTreeModel(TreeModel treeModel, Tree tree, boolean corrected, boolean nodeSampleCorrected){
		Visitor visitor = new AbstractVisitor(){

			private int[] nodeSamples = tree.getNodeSamples();


			@Override
			public VisitorAction visit(Node node){

				if(node.hasScore()){
					double nodeDepth = 0d;

					Deque<PMMLObject> parents = getParents();
					for(PMMLObject parent : parents){

						if(!(parent instanceof Node)){
							break;
						}

						nodeDepth++;
					}

					double nodeSample = this.nodeSamples[ValueUtil.asInt((Number)node.getId())];

					double averagePathLength = (corrected ? correctedAveragePathLength(nodeSample, nodeSampleCorrected) : averagePathLength(nodeSample));

					node.setScore(nodeDepth + averagePathLength);
				}

				return super.visit(node);
			}
		};
		visitor.applyTo(treeModel);
	}

	static
	private double averagePathLength(double n){

		if(n <= 1d){
			return 1d;
		}

		return 2d * (Math.log(n) + 0.5772156649) - 2d * ((n - 1d) / n);
	}

	static
	private double correctedAveragePathLength(double n, boolean nodeSampleCorrected){

		if(nodeSampleCorrected){

			if(n <= 1d){
				return 0d;
			} else

			if(n <= 2d){
				return 1d;
			}
		} else

		{
			if(n <= 1d){
				return 1d;
			}
		}

		return 2d * (Math.log(n - 1d) + 0.577215664901532860606512090082402431d) - 2d * ((n - 1d) / n);
	}
}