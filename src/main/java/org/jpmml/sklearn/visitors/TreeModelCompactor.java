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
package org.jpmml.sklearn.visitors;

import java.util.List;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;

public class TreeModelCompactor extends AbstractTreeModelTransformer {

	private MiningFunction miningFunction = null;


	@Override
	public void enterNode(Node node){
		Object id = node.getId();
		Object score = node.getScore();

		if(id == null){
			throw new IllegalArgumentException();
		} // End if

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			if(children.size() != 2){
				throw new IllegalArgumentException();
			}

			Node firstChild = children.get(0);
			Node secondChild = children.get(1);

			Predicate firstPredicate = firstChild.getPredicate();
			Predicate secondPredicate = secondChild.getPredicate();

			checkFieldReference(firstPredicate, secondPredicate);

			if(firstPredicate instanceof SimplePredicate && secondPredicate instanceof SimplePredicate){
				checkValue(firstPredicate, secondPredicate);
			} else

			{
				throw new IllegalArgumentException();
			} // End if

			if(hasOperator(firstPredicate, SimplePredicate.Operator.NOT_EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.EQUAL)){
				children = swapChildren(node);

				firstChild = children.get(0);
				secondChild = children.get(1);
			} else

			if(hasOperator(firstPredicate, SimplePredicate.Operator.LESS_OR_EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.GREATER_THAN)){
				// Ignored
			} else

			{
				throw new IllegalArgumentException();
			}

			secondChild.setPredicate(True.INSTANCE);
		} else

		{
			if(score == null){
				throw new IllegalArgumentException();
			}
		}

		node.setId(null);
	}

	@Override
	public void exitNode(Node node){
		Predicate predicate = node.getPredicate();

		if(predicate instanceof True){
			Node parentNode = getParentNode();

			if(parentNode == null){
				return;
			}

			if((MiningFunction.REGRESSION).equals(this.miningFunction)){
				parentNode.setScore(null);

				initScore(parentNode, node);
				replaceChildWithGrandchildren(parentNode, node);
			} else

			if((MiningFunction.CLASSIFICATION).equals(this.miningFunction)){

				// Replace intermediate nodes, but not terminal nodes
				if(node.hasNodes()){
					replaceChildWithGrandchildren(parentNode, node);
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	@Override
	public void enterTreeModel(TreeModel treeModel){
		TreeModel.MissingValueStrategy missingValueStrategy = treeModel.getMissingValueStrategy();
		TreeModel.NoTrueChildStrategy noTrueChildStrategy = treeModel.getNoTrueChildStrategy();
		TreeModel.SplitCharacteristic splitCharacteristic = treeModel.getSplitCharacteristic();

		if(!(TreeModel.MissingValueStrategy.NONE).equals(missingValueStrategy) || !(TreeModel.NoTrueChildStrategy.RETURN_NULL_PREDICTION).equals(noTrueChildStrategy) || !(TreeModel.SplitCharacteristic.BINARY_SPLIT).equals(splitCharacteristic)){
			throw new IllegalArgumentException();
		}

		this.miningFunction = treeModel.getMiningFunction();
	}

	@Override
	public void exitTreeModel(TreeModel treeModel){
		treeModel
			.setMissingValueStrategy(TreeModel.MissingValueStrategy.NULL_PREDICTION)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

		switch(this.miningFunction){
			case REGRESSION:
				treeModel.setNoTrueChildStrategy(TreeModel.NoTrueChildStrategy.RETURN_LAST_PREDICTION);
				break;
			case CLASSIFICATION:
				break;
			default:
				throw new IllegalArgumentException();
		}

		this.miningFunction = null;
	}
}