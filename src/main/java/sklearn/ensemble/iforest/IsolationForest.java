/*
 * Copyright (c) 2016 Villu Ruusmann
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

import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import numpy.core.Scalar;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import sklearn.Regressor;
import sklearn.ensemble.EnsembleRegressor;
import sklearn.tree.ExtraTreeRegressor;
import sklearn.tree.Tree;
import sklearn.tree.TreeModelUtil;

public class IsolationForest extends EnsembleRegressor {

	public IsolationForest(String module, String name){
		super(module, name);
	}

	@Override
	public boolean isSupervised(){
		return false;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();

		Schema segmentSchema = schema.toAnonymousSchema();

		List<TreeModel> treeModels = new ArrayList<>();

		for(Regressor estimator : estimators){
			ExtraTreeRegressor treeRegressor = (ExtraTreeRegressor)estimator;

			final
			Tree tree = treeRegressor.getTree();

			TreeModel treeModel = TreeModelUtil.encodeTreeModel(treeRegressor, MiningFunction.REGRESSION, segmentSchema);

			Visitor visitor = new AbstractVisitor(){

				private int[] nodeSamples = tree.getNodeSamples();


				@Override
				public VisitorAction visit(Node node){

					if(node.getScore() != null){
						double nodeDepth = 0d;

						Deque<PMMLObject> parents = getParents();
						for(PMMLObject parent : parents){

							if(!(parent instanceof Node)){
								break;
							}

							nodeDepth++;
						}

						double nodeSample = this.nodeSamples[Integer.parseInt(node.getId()) - 1];

						node.setScore(ValueUtil.formatValue(nodeDepth + averagePathLength(nodeSample)));
					}

					return super.visit(node);
				}
			};
			visitor.applyTo(treeModel);

			treeModels.add(treeModel);
		}

		OutputField rawAnomalyScore = new OutputField(FieldName.create("rawAnomalyScore"), DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setResultFeature(ResultFeature.PREDICTED_VALUE)
			.setFinalResult(false);

		OutputField normalizedAnomalyScore = new OutputField(FieldName.create("normalizedAnomalyScore"), DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setFinalResult(false)
			.setExpression(PMMLUtil.createApply("/", new FieldRef(rawAnomalyScore.getName()), PMMLUtil.createConstant(averagePathLength(getMaxSamples()))));

		// "0.5 - 2 ^ (-1 * normalizedAnomalyScore)"
		OutputField decisionFunction = new OutputField(FieldName.create("decisionFunction"), DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setFinalResult(false)
			.setExpression(PMMLUtil.createApply("-", PMMLUtil.createConstant(0.5d), PMMLUtil.createApply("pow", PMMLUtil.createConstant(2d), PMMLUtil.createApply("*", PMMLUtil.createConstant(-1d), new FieldRef(normalizedAnomalyScore.getName())))));

		OutputField outlier = new OutputField(FieldName.create("outlier"), DataType.BOOLEAN)
			.setOpType(OpType.CATEGORICAL)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(PMMLUtil.createApply("lessOrEqual", new FieldRef(decisionFunction.getName()), PMMLUtil.createConstant(getThreshold())));

		Output output = new Output()
			.addOutputFields(rawAnomalyScore, normalizedAnomalyScore, decisionFunction, outlier);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.AVERAGE, treeModels))
			.setOutput(output);

		return miningModel;
	}

	public int getMaxSamples(){
		return ValueUtil.asInt((Number)get("max_samples_"));
	}

	public double getThreshold(){
		Scalar threshold = (Scalar)get("threshold_");

		List<?> content = threshold.getContent();

		return ValueUtil.asDouble((Number)content.get(0));
	}

	static
	private double averagePathLength(double n){

		if(n <= 1d){
			return 1d;
		}

		return 2d * (Math.log(n) + 0.5772156649) - 2d * ((n - 1d) / n);
	}
}