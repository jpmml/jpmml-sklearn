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
package sklearn2pmml.tree;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.FieldNames;
import org.jpmml.converter.Schema;
import sklearn.HasApplyField;
import sklearn.Regressor;
import treelib.Tree;

public class CHAIDRegressor extends Regressor implements HasApplyField {

	public CHAIDRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public String getApplyField(){
		return FieldNames.NODE_ID;
	}

	@Override
	public TreeModel encodeModel(Schema schema){
		Tree tree = getTree();

		TreeModel treeModel = CHAIDUtil.encodeModel(MiningFunction.REGRESSION, tree, schema);

		encodeApplyOutput(treeModel, DataType.INTEGER);

		return treeModel;
	}

	public Tree getTree(){
		return get("treelib_tree_", Tree.class);
	}
}