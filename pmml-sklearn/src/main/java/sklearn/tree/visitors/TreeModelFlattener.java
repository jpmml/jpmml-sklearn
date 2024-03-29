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

import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;
import org.jpmml.model.UnsupportedAttributeException;
import org.jpmml.model.UnsupportedElementException;

public class TreeModelFlattener extends AbstractTreeModelTransformer {

	private MiningFunction miningFunction = null;


	@Override
	public void enterNode(Node node){

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			children:
			while(true){
				ListIterator<Node> childIt = children.listIterator();

				grandChildren:
				while(childIt.hasNext()){
					Node child = childIt.next();

					Iterator<Node> grandChildIt = getChildren(child);
					if(grandChildIt == null){
						continue grandChildren;
					}

					childIt.remove();

					while(grandChildIt.hasNext()){
						Node grandChild = grandChildIt.next();

						grandChildIt.remove();

						childIt.add(grandChild);
					}

					childIt.add(child);

					continue children;
				}

				break;
			}
		}
	}

	@Override
	public void exitNode(Node node){
		Predicate predicate = node.requirePredicate();

		if(predicate instanceof True){
			Node parentNode = getParentNode();

			if(parentNode == null){
				return;
			}

			List<Node> parentChildren = parentNode.getNodes();
			if(parentChildren.size() != 1){
				return;
			}

			boolean success = parentChildren.remove(node);
			if(!success){
				throw new UnsupportedElementException(parentNode);
			} // End if

			if(this.miningFunction == MiningFunction.CLASSIFICATION){
				initScoreDistribution(parentNode, node);
			} else

			if(this.miningFunction == MiningFunction.REGRESSION){
				parentNode.setScore(null);

				initScore(parentNode, node);
			}
		}
	}

	@Override
	public void enterTreeModel(TreeModel treeModel){
		super.enterTreeModel(treeModel);

		treeModel.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

		MiningFunction miningFunction = treeModel.requireMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
			case REGRESSION:
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

	static
	private Iterator<Node> getChildren(Node node){
		Predicate predicate = node.requirePredicate();

		if(!(predicate instanceof SimplePredicate)){
			return null;
		}

		SimplePredicate simplePredicate = (SimplePredicate)predicate;

		if(!hasOperator(simplePredicate, SimplePredicate.Operator.LESS_OR_EQUAL)){
			return null;
		} // End if

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			int endPos = 0;

			for(Node child : children){
				Predicate childPredicate = child.requirePredicate();

				if(!hasFieldReference(childPredicate, simplePredicate.requireField()) || !hasOperator(childPredicate, simplePredicate.requireOperator())){
					break;
				}

				endPos++;
			}

			if(endPos > 0){
				return (children.subList(0, endPos)).iterator();
			}

			return null;
		}

		return null;
	}
}