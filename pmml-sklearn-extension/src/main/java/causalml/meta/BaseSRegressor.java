/*
 * Copyright (c) 2026 Villu Ruusmann
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
package causalml.meta;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import causalml.CausalMLUtil;
import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;
import sklearn.EstimatorCastException;
import sklearn.EstimatorUtil;
import sklearn.Regressor;
import sklearn.ensemble.forest.ForestRegressor;
import sklearn.ensemble.gradient_boosting.GradientBoostingRegressor;
import sklearn.tree.TreeRegressor;

public class BaseSRegressor extends BaseSLearner<Regressor> {

	public BaseSRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Class<Regressor> getEstimatorClass(){
		return Regressor.class;
	}

	@Override
	public Model encodeEstimator(Role role, Regressor regressor, Schema schema){

		if(regressor instanceof TreeRegressor){
			TreeRegressor treeRegressor = (TreeRegressor)regressor;
		} else

		if(regressor instanceof ForestRegressor){
			ForestRegressor forestRegressor = (ForestRegressor)regressor;
		} else

		if(regressor instanceof GradientBoostingRegressor){
			GradientBoostingRegressor gradientBoostingRegressor = (GradientBoostingRegressor)regressor;
		} else

		{
			throw new EstimatorCastException(regressor, Arrays.asList(TreeRegressor.class, ForestRegressor.class, GradientBoostingRegressor.class));
		}

		Schema regressorSchema = CausalMLUtil.toRegressorSchema(regressor, schema);

		return EstimatorUtil.encodeNativeLike(regressor, regressorSchema);
	}

	@Override
	protected Model preOptimizeModel(Model model, String groupName){
		Visitor nullBranchMarker = new AbstractTreeModelTransformer(){

			private Set<Node> vitalNodes = new HashSet<>();


			@Override
			public void enterNode(Node node){
				Number recordCount = node.getRecordCount();

				if(recordCount != null){
					node.setRecordCount(null);
				}
			}

			@Override
			public void exitNode(Node node){
				Predicate predicate = node.requirePredicate();

				if(hasFieldReference(predicate, groupName)){
					this.vitalNodes.add(node);
				} // End if

				if(this.vitalNodes.contains(node)){
					Node parentNode = getParentNode();

					if(parentNode != null){
						this.vitalNodes.add(parentNode);
					}

					return;
				}

				Node treatmentAncestor = getAncestorNode(parentNode -> hasFieldReference(parentNode.requirePredicate(), groupName));
				if(treatmentAncestor == null){
					node.setScore(0d);
				}
			}

			@Override
			public void enterTreeModel(TreeModel treeModel){
				super.enterTreeModel(treeModel);
			}

			@Override
			public void exitTreeModel(TreeModel treeModel){
				super.exitTreeModel(treeModel);

				this.vitalNodes.clear();
			}
		};
		nullBranchMarker.applyTo(model);

		Visitor branchPruner = new AbstractTreeModelTransformer(){

			@Override
			public void exitNode(Node node){
				Object defaultChild = node.getDefaultChild();
				Object score = node.getScore();

				if(node.hasNodes()){
					List<Node> children = node.getNodes();

					Object childScore = getConstantScore(children);
					if(Objects.equals(score, childScore)){

						if(defaultChild != null){
							node.setDefaultChild(null);
						}

						children.clear();
					}
				}
			}

			private Object getConstantScore(List<Node> nodes){
				Object result = null;

				for(Node node : nodes){
					Object score = node.getScore();

					if(node.hasNodes()){
						return null;
					} // End if

					if(score == null){
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
		};
		branchPruner.applyTo(model);

		return model;
	}

	@Override
	protected Model postOptimizeModel(Model model, String groupName){
		return super.postOptimizeModel(model, groupName);
	}
}