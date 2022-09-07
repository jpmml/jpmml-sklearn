/*
 * Copyright (c) 2022 Villu Ruusmann
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
import java.util.Objects;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;
import org.jpmml.model.UnsupportedAttributeException;

public class TreeModelPruner extends AbstractTreeModelTransformer {

	private MiningFunction miningFunction = null;


	@Override
	public VisitorAction visit(Node node){

		if(node.hasScoreDistributions()){
			return VisitorAction.SKIP;
		}

		return super.visit(node);
	}

	@Override
	public void enterNode(Node node){
		super.enterNode(node);
	}

	@Override
	public void exitNode(Node node){
		Object score = node.getScore();

		if(node.hasScoreDistributions()){
			return;
		} // End if

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			Object childScore = getConstantScore(children);
			if(childScore != null){

				if(score == null){
					node.setScore(childScore);
				} // End if

				if(Objects.equals(score, childScore)){
					children.clear();
				}
			}
		}
	}

	@Override
	public void enterTreeModel(TreeModel treeModel){
		super.enterTreeModel(treeModel);

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
	private Object getConstantScore(List<Node> nodes){
		Object result = null;

		for(Node node : nodes){
			Object score = node.getScore();

			if(score == null || node.hasScoreDistributions()){
				return null;
			} // End if

			if(result == null){
				result = score;
			} else

			{
				if(!Objects.equals(score, result)){
					return null;
				}
			}
		}

		return result;
	}
}