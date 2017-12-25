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

import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.model.visitors.AbstractVisitor;

public class TreeModelFlattener extends AbstractVisitor {

	private MiningFunction miningFunction = null;


	@Override
	public void pushParent(PMMLObject object){
		super.pushParent(object);

		if(object instanceof TreeModel){
			handleTreeModelPush((TreeModel)object);
		}
	}

	@Override
	public PMMLObject popParent(){
		PMMLObject object = super.popParent();

		if(object instanceof Node){
			handleNodePop((Node)object);
		} else

		if(object instanceof TreeModel){
			handleTreeModelPop((TreeModel)object);
		}

		return object;
	}

	@Override
	public VisitorAction visit(Node node){

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

		return super.visit(node);
	}

	@Override
	public VisitorAction visit(TreeModel treeModel){
		treeModel.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

		return super.visit(treeModel);
	}

	private void handleNodePop(Node node){
		String score = node.getScore();
		Predicate predicate = node.getPredicate();

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
				throw new IllegalArgumentException();
			}

			String parentScore = parentNode.getScore();
			if(parentScore != null){
				throw new IllegalArgumentException();
			}

			parentNode.setScore(score);

			if((MiningFunction.REGRESSION).equals(this.miningFunction)){
				// Ignored
			} else

			if((MiningFunction.CLASSIFICATION).equals(this.miningFunction)){
				Double recordCount = node.getRecordCount();
				List<ScoreDistribution> scoreDistributions = node.getScoreDistributions();

				Double parentRecordCount = parentNode.getRecordCount();
				if(parentRecordCount != null){
					throw new IllegalArgumentException();
				}

				parentNode.setRecordCount(recordCount);

				List<ScoreDistribution> parentScoreDistributions = parentNode.getScoreDistributions();
				if(parentScoreDistributions.size() != 0){
					throw new IllegalArgumentException();
				}

				parentScoreDistributions.addAll(scoreDistributions);
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	private void handleTreeModelPush(TreeModel treeModel){
		this.miningFunction = treeModel.getMiningFunction();
	}

	private void handleTreeModelPop(TreeModel treeModel){
		this.miningFunction = null;
	}

	private Node getParentNode(){
		Deque<PMMLObject> parents = getParents();

		PMMLObject parent = parents.peekFirst();
		if(parent instanceof Node){
			return (Node)parent;
		}

		return null;
	}

	static
	private Iterator<Node> getChildren(Node node){
		Predicate predicate = node.getPredicate();

		if(!(predicate instanceof SimplePredicate)){
			return null;
		}

		SimplePredicate simplePredicate = (SimplePredicate)predicate;

		FieldName name = simplePredicate.getField();
		SimplePredicate.Operator operator = simplePredicate.getOperator();

		if(!(SimplePredicate.Operator.LESS_OR_EQUAL).equals(operator)){
			return null;
		} // End if

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			int endPos = 0;

			for(Node child : children){

				if(!checkPredicate(child, name, operator)){
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

	static
	private boolean checkPredicate(Node node, FieldName name, SimplePredicate.Operator operator){
		Predicate predicate = node.getPredicate();

		if(predicate instanceof SimplePredicate){
			SimplePredicate simplePredicate = (SimplePredicate)predicate;

			return (simplePredicate.getField()).equals(name) && (simplePredicate.getOperator()).equals(operator);
		}

		return false;
	}
}