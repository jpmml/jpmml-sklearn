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

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import sklearn.Estimator;

public class TreeModelUtil {

	private TreeModelUtil(){
	}

	static
	public <E extends Estimator & HasTree> List<TreeModel> encodeTreeModelSegmentation(List<E> estimators, MiningFunction miningFunction, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeTreeModelSegmentation(estimators, predicateManager, miningFunction, schema);
	}

	static
	public <E extends Estimator & HasTree> List<TreeModel> encodeTreeModelSegmentation(List<E> estimators, final PredicateManager predicateManager, final MiningFunction miningFunction, final Schema schema){
		final
		Schema segmentSchema = schema.toAnonymousSchema();

		Function<E, TreeModel> function = new Function<E, TreeModel>(){

			@Override
			public TreeModel apply(E estimator){
				Schema treeModelSchema = toTreeModelSchema(estimator.getDataType(), segmentSchema);

				return TreeModelUtil.encodeTreeModel(estimator, predicateManager, miningFunction, treeModelSchema);
			}
		};

		return new ArrayList<>(Lists.transform(estimators, function));
	}

	static
	public <E extends Estimator & HasTree> TreeModel encodeTreeModel(E estimator, MiningFunction miningFunction, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeTreeModel(estimator, predicateManager, miningFunction, schema);
	}

	static
	public <E extends Estimator & HasTree> TreeModel encodeTreeModel(E estimator, PredicateManager predicateManager, MiningFunction miningFunction, Schema schema){
		Tree tree = estimator.getTree();

		int[] leftChildren = tree.getChildrenLeft();
		int[] rightChildren = tree.getChildrenRight();
		int[] features = tree.getFeature();
		double[] thresholds = tree.getThreshold();
		double[] values = tree.getValues();

		Node root = new Node()
			.setId("1")
			.setPredicate(new True());

		encodeNode(root, predicateManager, 0, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

		TreeModel treeModel = new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema.getLabel()), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT);

		return treeModel;
	}

	static
	private void encodeNode(Node node, PredicateManager predicateManager, int index, int[] leftChildren, int[] rightChildren, int[] features, double[] thresholds, double[] values, MiningFunction miningFunction, Schema schema){
		int featureIndex = features[index];

		// A non-leaf (binary split) node
		if(featureIndex >= 0){
			Feature feature = schema.getFeature(featureIndex);

			double threshold = thresholds[index];

			Predicate leftPredicate;
			Predicate rightPredicate;

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				if(threshold < 0 || threshold > 1){
					throw new IllegalArgumentException();
				}

				String value = binaryFeature.getValue();

				leftPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.NOT_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.EQUAL, value);
			} else

			{
				ContinuousFeature continuousFeature = feature
					.toContinuousFeature(DataType.FLOAT) // First, cast from any numeric type (including numpy.float64) to numpy.float32
					.toContinuousFeature(DataType.DOUBLE); // Second, cast from numpy.float32 to numpy.float64

				String value = ValueUtil.formatValue(threshold);

				leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_OR_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_THAN, value);
			}

			int leftIndex = leftChildren[index];
			int rightIndex = rightChildren[index];

			Node leftChild = new Node()
				.setId(String.valueOf(leftIndex + 1))
				.setPredicate(leftPredicate);

			encodeNode(leftChild, predicateManager, leftIndex, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

			Node rightChild = new Node()
				.setId(String.valueOf(rightIndex + 1))
				.setPredicate(rightPredicate);

			encodeNode(rightChild, predicateManager, rightIndex, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

			node.addNodes(leftChild, rightChild);
		} else

		// A leaf node
		{
			if((MiningFunction.CLASSIFICATION).equals(miningFunction)){
				CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

				double[] scoreRecordCounts = getRow(values, leftChildren.length, categoricalLabel.size(), index);

				double recordCount = 0;

				for(double scoreRecordCount : scoreRecordCounts){
					recordCount += scoreRecordCount;
				}

				node.setRecordCount(recordCount);

				String score = null;

				Double probability = null;

				for(int i = 0; i < categoricalLabel.size(); i++){
					String value = categoricalLabel.getValue(i);

					ScoreDistribution scoreDistribution = new ScoreDistribution(value, scoreRecordCounts[i]);

					node.addScoreDistributions(scoreDistribution);

					double scoreProbability = (scoreRecordCounts[i] / recordCount);

					if(probability == null || probability.compareTo(scoreProbability) < 0){
						score = scoreDistribution.getValue();

						probability = scoreProbability;
					}
				}

				node.setScore(score);
			} else

			if((MiningFunction.REGRESSION).equals(miningFunction)){
				String score = ValueUtil.formatValue(values[index]);

				node.setScore(score);
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	static
	public Schema toTreeModelSchema(final DataType dataType, Schema schema){
		Function<Feature, Feature> function = new Function<Feature, Feature>(){

			@Override
			public Feature apply(Feature feature){

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					return binaryFeature;
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature(dataType);

					return continuousFeature;
				}
			}
		};

		return schema.toTransformedSchema(function);
	}

	static
	private double[] getRow(double[] values, int rows, int columns, int row){

		if(values.length != (rows * columns)){
			throw new IllegalArgumentException("Expected " + (rows * columns) + " element(s), got " + values.length + " element(s)");
		}

		double[] result = new double[columns];

		System.arraycopy(values, (row * columns), result, 0, columns);

		return result;
	}
}
