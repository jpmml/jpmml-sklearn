/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.tree;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Node;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.TreeModel;
import org.dmg.pmml.TreeModel.SplitCharacteristic;
import org.dmg.pmml.True;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ValueUtil;
import sklearn.Estimator;
import sklearn.EstimatorUtil;

public class TreeModelUtil {

	private TreeModelUtil(){
	}

	static
	public <E extends Estimator & HasTree> List<TreeModel> encodeTreeModelSegmentation(List<E> estimators, final MiningFunctionType miningFunction, final FeatureSchema schema){
		Function<E, TreeModel> function = new Function<E, TreeModel>(){

			private FeatureSchema segmentSchema = EstimatorUtil.createSegmentSchema(schema);


			@Override
			public TreeModel apply(E estimator){
				return TreeModelUtil.encodeTreeModel(estimator, miningFunction, this.segmentSchema);
			}
		};

		return Lists.newArrayList(Lists.transform(estimators, function));
	}

	static
	public <E extends Estimator & HasTree> TreeModel encodeTreeModel(E estimator, MiningFunctionType miningFunction, FeatureSchema schema){
		Tree tree = estimator.getTree();

		int[] leftChildren = tree.getChildrenLeft();
		int[] rightChildren = tree.getChildrenRight();
		int[] features = tree.getFeature();
		double[] thresholds = tree.getThreshold();
		double[] values = tree.getValues();

		Node root = new Node()
			.setId("1")
			.setPredicate(new True());

		encodeNode(root, 0, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

		MiningSchema miningSchema = ModelUtil.createMiningSchema(schema, root);

		TreeModel treeModel = new TreeModel(miningFunction, miningSchema, root)
			.setSplitCharacteristic(SplitCharacteristic.BINARY_SPLIT);

		return treeModel;
	}

	static
	private void encodeNode(Node node, int index, int[] leftChildren, int[] rightChildren, int[] features, double[] thresholds, double[] values, MiningFunctionType miningFunction, FeatureSchema schema){
		int featureIndex = features[index];

		// A non-leaf (binary split) node
		if(featureIndex >= 0){
			Feature feature = schema.getFeature(featureIndex);

			float threshold = (float)thresholds[index];

			Predicate leftPredicate;
			Predicate rightPredicate;

			if(feature instanceof ContinuousFeature){
				ContinuousFeature continuousFeature = (ContinuousFeature)feature;

				String value = ValueUtil.formatValue(threshold);

				leftPredicate = new SimplePredicate(continuousFeature.getName(), SimplePredicate.Operator.LESS_OR_EQUAL)
					.setValue(value);

				rightPredicate = new SimplePredicate(continuousFeature.getName(), SimplePredicate.Operator.GREATER_THAN)
					.setValue(value);
			} else

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				if(threshold < 0 || threshold > 1){
					throw new IllegalArgumentException();
				}

				leftPredicate = new SimplePredicate(binaryFeature.getName(), SimplePredicate.Operator.NOT_EQUAL)
					.setValue(binaryFeature.getValue());

				rightPredicate = new SimplePredicate(binaryFeature.getName(), SimplePredicate.Operator.EQUAL)
					.setValue(binaryFeature.getValue());
			} else

			{
				throw new IllegalArgumentException();
			}

			int leftIndex = leftChildren[index];
			int rightIndex = rightChildren[index];

			Node leftChild = new Node()
				.setId(String.valueOf(leftIndex + 1))
				.setPredicate(leftPredicate);

			encodeNode(leftChild, leftIndex, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

			Node rightChild = new Node()
				.setId(String.valueOf(rightIndex + 1))
				.setPredicate(rightPredicate);

			encodeNode(rightChild, rightIndex, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

			node.addNodes(leftChild, rightChild);
		} else

		// A leaf node
		{
			if((MiningFunctionType.CLASSIFICATION).equals(miningFunction)){
				List<String> targetCategories = schema.getTargetCategories();

				double[] scoreRecordCounts = getRow(values, leftChildren.length, targetCategories.size(), index);

				double recordCount = 0;

				for(double scoreRecordCount : scoreRecordCounts){
					recordCount += scoreRecordCount;
				}

				node.setRecordCount(recordCount);

				String score = null;

				Double probability = null;

				for(int i = 0; i < targetCategories.size(); i++){
					String targetCategory = targetCategories.get(i);

					ScoreDistribution scoreDistribution = new ScoreDistribution(targetCategory, scoreRecordCounts[i]);

					node.addScoreDistributions(scoreDistribution);

					double scoreProbability = (scoreRecordCounts[i] / recordCount);

					if(probability == null || probability.compareTo(scoreProbability) < 0){
						score = scoreDistribution.getValue();

						probability = scoreProbability;
					}
				}

				node.setScore(score);
			} else

			if((MiningFunctionType.REGRESSION).equals(miningFunction)){
				String score = ValueUtil.formatValue(values[index]);

				node.setScore(score);
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	static
	private double[] getRow(double[] values, int rows, int columns, int row){

		if(values.length != (rows * columns)){
			throw new IllegalArgumentException();
		}

		double[] result = new double[columns];

		System.arraycopy(values, (row * columns), result, 0, columns);

		return result;
	}
}