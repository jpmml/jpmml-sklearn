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
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import sklearn.Regressor;
import sklearn.tree.HasTree;
import sklearn.tree.TreeUtil;

public class ObliqueDecisionTreeRegressor extends Regressor implements HasTree {

	public ObliqueDecisionTreeRegressor(String module, String name){
		super(module, name);
	}

	public ObliqueDecisionTreeRegressor(ObliqueDecisionTreeRegressor that){
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
		ProjectionManager projectionManager = new ProjectionManager();

		return encodeModel(predicateManager, projectionManager, schema);
	}

	public TreeModel encodeModel(PredicateManager predicateManager, ProjectionManager projectionManager, Schema schema){
		ObliqueTree tree = getTree();

		if(tree.hasProjVecs()){
			Object segmentId = getPMMLSegmentId();

			Schema sklearnSchema = tree.transformSchema(segmentId, projectionManager, schema);

			ObliqueDecisionTreeRegressor sklearnRegressor = new ObliqueDecisionTreeRegressor(this){

				private ObliqueTree sklearnTree = tree.transform(sklearnSchema);


				@Override
				public ObliqueTree getTree(){
					return this.sklearnTree;
				}
			};

			return sklearnRegressor.encodeModel(predicateManager, projectionManager, sklearnSchema);
		}

		return TreeUtil.encodeTreeModel(this, MiningFunction.REGRESSION, predicateManager, null, schema);
	}

	@Override
	public ObliqueTree getTree(){
		return get("tree_", ObliqueTree.class);
	}

	@Override
	public boolean hasMissingValueSupport(){
		return false;
	}
}