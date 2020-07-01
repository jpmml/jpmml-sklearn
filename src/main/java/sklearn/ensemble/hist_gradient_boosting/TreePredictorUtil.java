/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.ensemble.hist_gradient_boosting;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.BranchNode;
import org.dmg.pmml.tree.LeafNode;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;

public class TreePredictorUtil {

	private TreePredictorUtil(){
	}

	static
	public TreeModel encodeTreeModel(TreePredictor treePredictor, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeTreeModel(treePredictor, predicateManager, schema);
	}

	static
	public TreeModel encodeTreeModel(TreePredictor treePredictor, PredicateManager predicateManager, Schema schema){
		int[] leaf = treePredictor.isLeaf();
		int[] leftChildren = treePredictor.getLeft();
		int[] rightChildren = treePredictor.getRight();
		int[] featureIdx = treePredictor.getFeatureIdx();
		double[] thresholds = treePredictor.getThreshold();
		int[] missingGoToLeft = treePredictor.getMissingGoToLeft();
		double[] values = treePredictor.getValues();

		Node root = encodeNode(0, True.INSTANCE, leaf, leftChildren, rightChildren, featureIdx, thresholds, missingGoToLeft, values, predicateManager, schema);

		TreeModel treeModel = new TreeModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT)
			.setMissingValueStrategy(TreeModel.MissingValueStrategy.DEFAULT_CHILD);

		return treeModel;
	}

	static
	private Node encodeNode(int index, Predicate predicate, int[] leaf, int[] leftChildren, int[] rightChildren, int[] featureIdx, double[] thresholds, int[] missingGoToLeft, double[] values, PredicateManager predicateManager, Schema schema){
		Integer id = Integer.valueOf(index);

		if(leaf[index] == 0){
			Feature feature = schema.getFeature(featureIdx[index]);

			double threshold = thresholds[index];

			Predicate leftPredicate;
			Predicate rightPredicate;

			boolean defaultLeft;

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				if(threshold != 0.5d){
					throw new IllegalArgumentException();
				}

				Object value = binaryFeature.getValue();

				leftPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.NOT_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.EQUAL, value);

				// XXX
				defaultLeft = true;
			} else

			{
				ContinuousFeature continuousFeature = feature.toContinuousFeature(DataType.DOUBLE);

				Double value = threshold;

				leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_OR_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_THAN, value);

				defaultLeft = (missingGoToLeft[index] == 1);
			}

			Node leftChild = encodeNode(leftChildren[index], leftPredicate, leaf, leftChildren, rightChildren, featureIdx, thresholds, missingGoToLeft, values, predicateManager, schema);
			Node rightChild = encodeNode(rightChildren[index], rightPredicate, leaf, leftChildren, rightChildren, featureIdx, thresholds, missingGoToLeft, values, predicateManager, schema);

			Node result = new BranchNode(null, predicate)
				.setId(id)
				.setDefaultChild(defaultLeft ? leftChild.getId() : rightChild.getId())
				.addNodes(leftChild, rightChild);

			return result;
		} else

		if(leaf[index] == 1){
			Node result = new LeafNode(values[index], predicate)
				.setId(id);

			return result;
		} else

		{
			throw new IllegalArgumentException();
		}
	}
}