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

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.False;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.BranchNode;
import org.dmg.pmml.tree.LeafNode;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoryManager;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.MissingValueFeature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaException;
import org.jpmml.sklearn.SkLearnException;

public class TreePredictorUtil {

	private TreePredictorUtil(){
	}

	static
	public TreeModel encodeTreeModel(TreePredictor treePredictor, BinMapper binMapper, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeTreeModel(treePredictor, binMapper, predicateManager, schema);
	}

	static
	public TreeModel encodeTreeModel(TreePredictor treePredictor, BinMapper binMapper, PredicateManager predicateManager, Schema schema){
		int[] leaf = treePredictor.isLeaf();
		int[] leftChildren = treePredictor.getLeft();
		int[] rightChildren = treePredictor.getRight();
		int[] featureIdx = treePredictor.getFeatureIdx();
		int[] isCategorical = treePredictor.isCategorical();
		double[] thresholds = treePredictor.getThreshold();
		int[] bitsetIdx = treePredictor.getBitsetIdx();
		int[] missingGoToLeft = treePredictor.getMissingGoToLeft();
		double[] values = treePredictor.getValues();
		int[] rawLeftCatBitsets = treePredictor.getRawLeftCatBitsets();

		Node root = encodeNode(0, True.INSTANCE, leaf, leftChildren, rightChildren, featureIdx, isCategorical, thresholds, bitsetIdx, missingGoToLeft, values, binMapper, rawLeftCatBitsets, new CategoryManager(), predicateManager, schema);

		TreeModel treeModel = new TreeModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT)
			.setMissingValueStrategy(TreeModel.MissingValueStrategy.DEFAULT_CHILD);

		return treeModel;
	}

	static
	private Node encodeNode(int index, Predicate predicate, int[] leaf, int[] leftChildren, int[] rightChildren, int[] featureIdx, int[] isCategorical, double[] thresholds, int[] bitsetIdx, int[] missingGoToLeft, double[] values, BinMapper binMapper, int[] rawLeftCatBitsets, CategoryManager categoryManager, PredicateManager predicateManager, Schema schema){
		Integer id = Integer.valueOf(index);

		if(leaf[index] == 0){
			Feature feature = schema.getFeature(featureIdx[index]);

			CategoryManager leftCategoryManager = categoryManager;
			CategoryManager rightCategoryManager = categoryManager;

			Predicate leftPredicate;
			Predicate rightPredicate;

			boolean defaultLeft = (missingGoToLeft[index] == 1);

			boolean categorical = ((isCategorical != null) && (isCategorical[index] == 1));
			if(categorical){

				if(feature instanceof CategoricalFeature){
					CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

					String name = categoricalFeature.getName();

					java.util.function.Predicate<Object> valueFilter = categoryManager.getValueFilter(name);

					int row = bitsetIdx[index];

					// XXX
					int rawLeftCatBitset = rawLeftCatBitsets[row * 8];

					List<Object> leftValues = new ArrayList<>();
					List<Object> rightValues = new ArrayList<>();

					for(int i = 0; i < categoricalFeature.size(); i++){
						Object value = categoricalFeature.getValue(i);

						if(!valueFilter.test(value)){
							continue;
						} // End if

						if(((rawLeftCatBitset >> i) & 1) == 1){
							leftValues.add(value);
						} else

						{
							rightValues.add(value);
						}
					}

					leftCategoryManager = categoryManager.fork(name, leftValues);
					rightCategoryManager = categoryManager.fork(name, rightValues);

					if(!leftValues.isEmpty()){
						leftPredicate = predicateManager.createPredicate(categoricalFeature, leftValues);
					} else

					{
						leftPredicate = False.INSTANCE;
					} // End if

					if(!rightValues.isEmpty()){
						rightPredicate = predicateManager.createPredicate(categoricalFeature, rightValues);
					} else

					{
						rightPredicate = False.INSTANCE;
					}
				} else

				{
					throw new SchemaException("Expected a categorical feature, got " + feature);
				}
			} else

			{
				double threshold = thresholds[index];

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					if(threshold != 0.5d){
						throw new SkLearnException("Expected 0.5 threshold value, got " + threshold);
					}

					Object value = binaryFeature.getValue();

					leftPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.NOT_EQUAL, value);
					rightPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.EQUAL, value);

					// XXX
					defaultLeft = true;
				} else

				if(feature instanceof MissingValueFeature){
					MissingValueFeature missingValueFeature = (MissingValueFeature)feature;

					if(threshold != 0.5d){
						throw new SkLearnException("Expected 0.5 threshold value, got " + threshold);
					}

					leftPredicate = predicateManager.createSimplePredicate(missingValueFeature, SimplePredicate.Operator.IS_NOT_MISSING, null);
					rightPredicate = predicateManager.createSimplePredicate(missingValueFeature, SimplePredicate.Operator.IS_MISSING, null);
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature(DataType.DOUBLE);

					Object value;

					if(threshold == Double.POSITIVE_INFINITY){
						value = "INF";
					} else

					{
						value = threshold;
					}

					leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_OR_EQUAL, value);
					rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_THAN, value);
				}
			}

			Node leftChild = encodeNode(leftChildren[index], leftPredicate, leaf, leftChildren, rightChildren, featureIdx, isCategorical, thresholds, bitsetIdx, missingGoToLeft, values, binMapper, rawLeftCatBitsets, leftCategoryManager, predicateManager, schema);
			Node rightChild = encodeNode(rightChildren[index], rightPredicate, leaf, leftChildren, rightChildren, featureIdx, isCategorical, thresholds, bitsetIdx, missingGoToLeft, values, binMapper, rawLeftCatBitsets, rightCategoryManager, predicateManager, schema);

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