/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.tree;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.FieldNames;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import sklearn.HasApplyField;
import sklearn.SkLearnRegressor;

public class TreeRegressor extends SkLearnRegressor implements HasApplyField, HasTree, HasTreeOptions {

	public TreeRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public String getApplyField(){
		return FieldNames.NODE_ID;
	}

	@Override
	public TreeModel encodeModel(Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeModel(predicateManager, schema);
	}

	public TreeModel encodeModel(PredicateManager predicateManager, Schema schema){
		return TreeUtil.encodeTreeModel(this, MiningFunction.REGRESSION, predicateManager, null, schema);
	}

	@Override
	public Schema configureSchema(Schema schema){
		return TreeUtil.configureSchema(this, schema);
	}

	@Override
	public Model configureModel(Model model){
		return TreeUtil.configureModel(this, model);
	}

	@Override
	public Tree getTree(){
		return get("tree_", Tree.class);
	}

	@Override
	public boolean hasMissingValueSupport(){
		return TreeUtil.hasMissingValueSupport(this);
	}
}