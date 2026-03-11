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
import java.util.List;

import causalml.CausalMLUtil;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;
import org.jpmml.model.UnsupportedElementException;
import sklearn.Classifier;
import sklearn.EstimatorCastException;
import sklearn.EstimatorUtil;
import sklearn.ensemble.forest.ForestClassifier;
import sklearn.tree.TreeClassifier;

public class BaseSClassifier extends BaseSLearner<Classifier> {

	public BaseSClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Class<Classifier> getEstimatorClass(){
		return Classifier.class;
	}

	@Override
	public Model encodeEstimator(Role role, Classifier classifier, Schema schema){

		if(classifier instanceof TreeClassifier){
			TreeClassifier treeClassifier = (TreeClassifier)classifier;
		} else

		if(classifier instanceof ForestClassifier){
			ForestClassifier forestClassifier = (ForestClassifier)classifier;
		} else

		{
			throw new EstimatorCastException(classifier, Arrays.asList(TreeClassifier.class, ForestClassifier.class));
		}

		Schema classifierSchema = CausalMLUtil.toClassifierSchema(classifier, schema);

		Model model = EstimatorUtil.encodeNativeLike(classifier, classifierSchema);

		Visitor visitor = new AbstractTreeModelTransformer(){

			@Override
			public void pushParent(PMMLObject object){
				super.pushParent(object);

				if(object instanceof MiningModel){
					enterMiningModel((MiningModel)object);
				}
			}

			@Override
			public PMMLObject popParent(){
				PMMLObject object = super.popParent();

				if(object instanceof MiningModel){
					exitMiningModel((MiningModel)object);
				}

				return object;
			}

			@Override
			public void enterNode(Node node){
			}

			@Override
			public void exitNode(Node node){
				Object score = node.getScore();

				if(node.hasNodes()){

					if(score != null || node.hasScoreDistributions()){
						throw new UnsupportedElementException(node);
					}

					node.setScore(0d);
				} else

				{
					if(!node.hasScoreDistributions()){
						throw new UnsupportedElementException(node);
					}

					Number recordCount = node.getRecordCount();
					List<ScoreDistribution> scoreDistributions = node.getScoreDistributions();

					if((recordCount.doubleValue() != 1d) || (scoreDistributions.size() != 2)){
						throw new UnsupportedElementException(node);
					}

					ScoreDistribution eventScoreDistribution = scoreDistributions.get(1);

					Number eventRecordCount = eventScoreDistribution.getRecordCount();
					if(eventRecordCount == null){
						throw new UnsupportedElementException(eventScoreDistribution);
					}

					node
						.setScore(eventRecordCount.doubleValue())
						.setRecordCount(null);

					scoreDistributions.clear();
				}
			}

			@Override
			public void enterTreeModel(TreeModel treeModel){
				super.enterTreeModel(treeModel);
			}

			@Override
			public void exitTreeModel(TreeModel treeModel){
				super.exitTreeModel(treeModel);

				treeModel
					.setMiningFunction(MiningFunction.REGRESSION)
					.setOutput(null);
			}

			public void enterMiningModel(MiningModel miningModel){
			}

			public void exitMiningModel(MiningModel miningModel){
				miningModel
					.setMiningFunction(MiningFunction.REGRESSION)
					.setOutput(null);
			}
		};
		visitor.applyTo(model);

		return model;
	}
}