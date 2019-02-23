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

import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.AbstractTransformation;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.OutlierTransformation;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnUtil;
import sklearn.Regressor;
import sklearn.ensemble.EnsembleRegressor;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.ScoreDistributionManager;
import sklearn.tree.Tree;
import sklearn.tree.TreeModelUtil;
import sklearn.tree.TreeRegressor;

public class IsolationForest extends EnsembleRegressor implements HasTreeOptions {

	public IsolationForest(String module, String name){
		super(module, name);
	}

	@Override
	public boolean isSupervised(){
		return false;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		String sklearnVersion = getSkLearnVersion();
		List<? extends Regressor> estimators = getEstimators();

		// See https://github.com/scikit-learn/scikit-learn/issues/8549
		boolean corrected = (sklearnVersion != null && SkLearnUtil.compareVersion(sklearnVersion, "0.19") >= 0);

		PredicateManager predicateManager = new PredicateManager();
		ScoreDistributionManager scoreDistributionManager = new ScoreDistributionManager();

		Schema segmentSchema = schema.toAnonymousSchema();

		List<TreeModel> treeModels = new ArrayList<>();

		for(Regressor estimator : estimators){
			TreeRegressor treeRegressor = (TreeRegressor)estimator;

			Tree tree = treeRegressor.getTree();

			TreeModel treeModel = TreeModelUtil.encodeTreeModel(treeRegressor, predicateManager, scoreDistributionManager, MiningFunction.REGRESSION, segmentSchema);

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

						double nodeSample = this.nodeSamples[Integer.parseInt(node.getId())];

						double averagePathLength = (corrected ? correctedAveragePathLength(nodeSample) : averagePathLength(nodeSample));

						node.setScore(nodeDepth + averagePathLength);
					}

					return super.visit(node);
				}
			};
			visitor.applyTo(treeModel);

			ClassDictUtil.clearContent(tree);

			treeModels.add(treeModel);
		}

		// "rawAnomalyScore / averagePathLength(maxSamples)"
		Transformation normalizedAnomalyScore = new AbstractTransformation(){

			@Override
			public FieldName getName(FieldName name){
				return FieldName.create("normalizedAnomalyScore");
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				double maxSamples = getMaxSamples();

				double averagePathLength = (corrected ? correctedAveragePathLength(maxSamples) : averagePathLength(maxSamples));

				return PMMLUtil.createApply("/", fieldRef, PMMLUtil.createConstant(averagePathLength));
			}
		};

		// "0.5 - 2 ^ (-1 * normalizedAnomalyScore)"
		Transformation decisionFunction = new AbstractTransformation(){

			@Override
			public FieldName getName(FieldName name){
				return FieldName.create("decisionFunction");
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				return PMMLUtil.createApply("-", PMMLUtil.createConstant(0.5d), PMMLUtil.createApply("pow", PMMLUtil.createConstant(2d), PMMLUtil.createApply("*", PMMLUtil.createConstant(-1d), fieldRef)));
			}
		};

		Transformation outlier = new OutlierTransformation(){

			@Override
			public Expression createExpression(FieldRef fieldRef){
				String behaviour = getBehaviour();

				double threshold;

				// SkLearn 0.19
				if(behaviour == null){
					threshold = getThreshold();
				} else

				// SkLearn 0.20+
				{
					if(("old").equals(behaviour)){
						threshold = getThreshold();
					} else

					{
						throw new IllegalArgumentException(behaviour);
					}
				}

				return PMMLUtil.createApply("lessOrEqual", fieldRef, PMMLUtil.createConstant(threshold));
			}
		};

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.AVERAGE, treeModels))
			.setOutput(ModelUtil.createPredictedOutput(FieldName.create("rawAnomalyScore"), OpType.CONTINUOUS, DataType.DOUBLE, normalizedAnomalyScore, decisionFunction, outlier));

		return TreeModelUtil.transform(this, miningModel);
	}

	public String getBehaviour(){
		return getOptionalString("behaviour");
	}

	public int getMaxSamples(){
		return ValueUtil.asInt(getNumber("max_samples_"));
	}

	public double getThreshold(){
		Number threshold;

		// SkLearn 0.19
		if(containsKey("threshold_")){
			threshold = getNumber("threshold_");
		} else

		// SkLearn 0.20+
		{
			threshold = getNumber("_threshold_");
		}

		return ValueUtil.asDouble(threshold);
	}

	static
	private double averagePathLength(double n){

		if(n <= 1d){
			return 1d;
		}

		return 2d * (Math.log(n) + 0.5772156649) - 2d * ((n - 1d) / n);
	}

	static
	private double correctedAveragePathLength(double n){

		if(n <= 1d){
			return 1d;
		}

		return 2d * (Math.log(n - 1d) + 0.577215664901532860606512090082402431d) - 2d * ((n - 1d) / n);
	}
}