/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sktree.tree;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ScoreDistributionManager;
import sklearn.Classifier;
import sklearn.tree.HasTree;
import sklearn.tree.TreeUtil;

public class ObliqueDecisionTreeClassifier extends Classifier implements HasTree {

	public ObliqueDecisionTreeClassifier(String module, String name){
		super(module, name);
	}

	public ObliqueDecisionTreeClassifier(ObliqueDecisionTreeClassifier that){
		this(that.getPythonModule(), that.getPythonName());

		update(that);
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public TreeModel encodeModel(Schema schema){
		PredicateManager predicateManager = new PredicateManager();
		ScoreDistributionManager scoreDistributionManager = new ScoreDistributionManager();
		ProjectionManager projectionManager = new ProjectionManager();

		return encodeModel(predicateManager, scoreDistributionManager, projectionManager, schema);
	}

	public TreeModel encodeModel(PredicateManager predicateManager, ScoreDistributionManager scoreDistributionManager, ProjectionManager projectionManager, Schema schema){
		ObliqueTree tree = getTree();

		if(tree.hasProjVecs()){
			Object segmentId = getPMMLSegmentId();

			Schema sklearnSchema = tree.transformSchema(segmentId, projectionManager, schema);

			ObliqueDecisionTreeClassifier sklearnClassifier = new ObliqueDecisionTreeClassifier(this){

				private ObliqueTree sklearnTree = tree.transform(sklearnSchema);


				@Override
				public ObliqueTree getTree(){
					return this.sklearnTree;
				}
			};

			return sklearnClassifier.encodeModel(predicateManager, scoreDistributionManager, projectionManager, sklearnSchema);
		}

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		TreeModel treeModel = TreeUtil.encodeTreeModel(this, MiningFunction.CLASSIFICATION, predicateManager, scoreDistributionManager, schema);

		encodePredictProbaOutput(treeModel, DataType.DOUBLE, categoricalLabel);

		return treeModel;
	}

	@Override
	public ObliqueTree getTree(){
		return get("tree_", ObliqueTree.class);
	}
}