/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.tree.visitors;

import java.util.List;

import org.dmg.pmml.HasFieldReference;
import org.dmg.pmml.HasValue;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;
import org.jpmml.model.UnsupportedAttributeException;
import org.jpmml.model.UnsupportedElementException;

public class TreeModelCompactor extends AbstractTreeModelTransformer {

	private MiningFunction miningFunction = null;


	@Override
	public void enterNode(Node node){
		Object id = node.getId();
		Object score = node.getScore();
		Object defaultChild = node.getDefaultChild();

		if(id == null){
			throw new UnsupportedElementException(node);
		} // End if

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			if(children.size() != 2){
				throw new UnsupportedElementException(node);
			}

			Node firstChild = children.get(0);
			Node secondChild = children.get(1);

			SimplePredicate firstPredicate = firstChild.requirePredicate(SimplePredicate.class);
			SimplePredicate secondPredicate = secondChild.requirePredicate(SimplePredicate.class);

			checkFieldReference((HasFieldReference<?>)firstPredicate, secondPredicate);
			checkValue((HasValue<?>)firstPredicate, secondPredicate);

			if(defaultChild != null){

				if(equalsNode(defaultChild, firstChild)){
					children = swapChildren(node);

					firstChild = children.get(0);
					secondChild = children.get(1);
				} else

				if(equalsNode(defaultChild, secondChild)){
					// Ignored
				} else

				{
					throw new UnsupportedElementException(node);
				}

				node.setDefaultChild(null);
			} else

			{
				if(hasOperator(firstPredicate, SimplePredicate.Operator.NOT_EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.EQUAL)){
					children = swapChildren(node);

					firstChild = children.get(0);
					secondChild = children.get(1);
				} else

				if(hasOperator(firstPredicate, SimplePredicate.Operator.LESS_OR_EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.GREATER_THAN)){
					// Ignored
				} else

				{
					throw new UnsupportedElementException(node);
				}
			}

			secondChild.setPredicate(True.INSTANCE);
		} else

		{
			if(score == null){
				throw new UnsupportedElementException(node);
			}
		}

		node.setId(null);
	}

	@Override
	public void exitNode(Node node){
		Predicate predicate = node.requirePredicate();

		if(predicate instanceof True){
			Node parentNode = getParentNode();

			if(parentNode == null){
				return;
			} // End if

			if(this.miningFunction == MiningFunction.CLASSIFICATION){

				// Replace intermediate nodes, but not terminal nodes
				if(node.hasNodes()){
					replaceChildWithGrandchildren(parentNode, node);
				}
			} else

			if(this.miningFunction == MiningFunction.REGRESSION){
				parentNode.setScore(null);

				initScore(parentNode, node);
				replaceChildWithGrandchildren(parentNode, node);
			}
		}
	}

	@Override
	public void enterTreeModel(TreeModel treeModel){
		super.enterTreeModel(treeModel);

		TreeModel.MissingValueStrategy missingValueStrategy = treeModel.getMissingValueStrategy();
		if(missingValueStrategy == TreeModel.MissingValueStrategy.NONE){
			// Missing values are not allowed
			missingValueStrategy = TreeModel.MissingValueStrategy.NULL_PREDICTION;
		} else

		if(missingValueStrategy == TreeModel.MissingValueStrategy.DEFAULT_CHILD){
			// Missing values are allowed
			missingValueStrategy = TreeModel.MissingValueStrategy.NONE;
		} else

		{
			throw new UnsupportedAttributeException(treeModel, missingValueStrategy);
		}

		TreeModel.NoTrueChildStrategy noTrueChildStrategy = treeModel.getNoTrueChildStrategy();
		if(noTrueChildStrategy != TreeModel.NoTrueChildStrategy.RETURN_NULL_PREDICTION){
			throw new UnsupportedAttributeException(treeModel, noTrueChildStrategy);
		}

		TreeModel.SplitCharacteristic splitCharacteristic = treeModel.getSplitCharacteristic();
		if(splitCharacteristic != TreeModel.SplitCharacteristic.BINARY_SPLIT){
			throw new UnsupportedAttributeException(treeModel, splitCharacteristic);
		}

		treeModel
			.setMissingValueStrategy(missingValueStrategy)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

		MiningFunction miningFunction = treeModel.requireMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
				break;
			case REGRESSION:
				treeModel.setNoTrueChildStrategy(TreeModel.NoTrueChildStrategy.RETURN_LAST_PREDICTION);
				break;
			default:
				throw new UnsupportedAttributeException(treeModel, miningFunction);
		}

		this.miningFunction = miningFunction;
	}

	@Override
	public void exitTreeModel(TreeModel treeModel){
		super.exitTreeModel(treeModel);

		this.miningFunction = null;
	}
}