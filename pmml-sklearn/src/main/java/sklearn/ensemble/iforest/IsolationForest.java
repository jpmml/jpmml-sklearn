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

import com.google.common.primitives.Ints;
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
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ScoreDistributionManager;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.transformations.AbstractTransformation;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.OutlierDetector;
import sklearn.OutlierDetectorUtil;
import sklearn.Regressor;
import sklearn.VersionUtil;
import sklearn.ensemble.EnsembleRegressor;
import sklearn.ensemble.EnsembleUtil;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.Tree;
import sklearn.tree.TreeRegressor;
import sklearn.tree.TreeUtil;

public class IsolationForest extends EnsembleRegressor implements HasTreeOptions, OutlierDetector {

	public IsolationForest(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfOutputs(){
		return 0;
	}

	@Override
	public boolean isSupervised(){
		return false;
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		throw new UnsupportedOperationException();
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		String sklearnVersion = getSkLearnVersion();
		List<? extends Regressor> estimators = getEstimators();
		List<List<Integer>> estimatorsFeatures = getEstimatorsFeatures();

		// See https://github.com/scikit-learn/scikit-learn/issues/8549
		boolean corrected = (sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "0.19") >= 0);

		// See https://github.com/scikit-learn/scikit-learn/issues/11839
		boolean nodeSampleCorrected = (sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "0.21") >= 0);

		Boolean numeric = (Boolean)getOption(HasTreeOptions.OPTION_NUMERIC, Boolean.TRUE);

		Schema treeModelSchema = TreeUtil.toTreeModelSchema(getDataType(), numeric, schema);

		PredicateManager predicateManager = new PredicateManager();
		ScoreDistributionManager scoreDistributionManager = new ScoreDistributionManager();

		Schema segmentSchema = treeModelSchema.toAnonymousSchema();

		List<TreeModel> treeModels = new ArrayList<>();

		for(int i = 0; i < estimators.size(); i++){
			Regressor estimator = estimators.get(i);
			List<Integer> estimatorFeatures = estimatorsFeatures.get(i);

			Schema estimatorSchema = segmentSchema.toSubSchema(Ints.toArray(estimatorFeatures));

			TreeRegressor treeRegressor = (TreeRegressor)estimator;

			Tree tree = treeRegressor.getTree();

			TreeModel treeModel = TreeUtil.encodeTreeModel(treeRegressor, MiningFunction.REGRESSION, predicateManager, scoreDistributionManager, estimatorSchema);

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

			ClassDictUtil.clearContent(tree);

			treeModels.add(treeModel);
		}

		// "rawAnomalyScore / averagePathLength(maxSamples)"
		Transformation normalizedAnomalyScore = new AbstractTransformation(){

			@Override
			public String getName(String name){
				return "normalizedAnomalyScore";
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				double maxSamples = getMaxSamples();

				double averagePathLength = (corrected ? correctedAveragePathLength(maxSamples, nodeSampleCorrected) : averagePathLength(maxSamples));

				return ExpressionUtil.createApply(PMMLFunctions.DIVIDE, fieldRef, ExpressionUtil.createConstant(averagePathLength));
			}
		};

		// "0.5 - 2 ^ (-1 * normalizedAnomalyScore)"
		Transformation decisionFunction = new AbstractTransformation(){

			@Override
			public String getName(String name){
				return getDecisionFunctionField();
			}

			@Override
			public boolean isFinalResult(){
				return true;
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				Number offset = getOffset();

				return ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, ExpressionUtil.createConstant(-offset.doubleValue()), ExpressionUtil.createApply(PMMLFunctions.POW, ExpressionUtil.createConstant(2d), ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, ExpressionUtil.createConstant(-1d), fieldRef)));
			}
		};

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.AVERAGE, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels))
			.setOutput(OutlierDetectorUtil.createPredictedOutput(this, "rawAnomalyScore", normalizedAnomalyScore, decisionFunction));

		return TreeUtil.transform(this, miningModel);
	}

	@Override
	public Number getDecisionFunctionThreshold(){
		String behaviour = getBehaviour();

		Number threshold;

		// SkLearn 0.19 or SkLearn 0.24+
		if(behaviour == null){
			threshold = getThreshold();
		} else

		// SkLearn 0.20 through 0.23
		{
			if(("old").equals(behaviour)){
				threshold = getThreshold();
			} else

			if(("new").equals(behaviour) || ("deprecated").equals(behaviour)){
				threshold = 0d;
			} else

			{
				throw new IllegalArgumentException(behaviour);
			}
		}

		return threshold;
	}

	public List<List<Integer>> getEstimatorsFeatures(){
		return EnsembleUtil.transformEstimatorsFeatures(getList("estimators_features_", HasArray.class));
	}

	public String getBehaviour(){
		return getOptionalString("behaviour");
	}

	public int getMaxSamples(){
		return getInteger("max_samples_");
	}

	public Number getOffset(){

		if(!containsKey("offset_")){
			return 0.5d;
		}

		return getNumber("offset_");
	}

	public Number getThreshold(){

		// SkLearn 0.19
		if(containsKey("threshold_")){
			return getNumber("threshold_");
		} else

		// SkLearn 0.20+
		if(containsKey("_threshold_")){
			return getNumber("_threshold_");
		}

		// SkLearn 0.24+
		return 0d;
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