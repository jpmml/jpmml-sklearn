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

import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.Output;
import org.dmg.pmml.TreeModel;
import org.jpmml.converter.PMMLUtil;
import sklearn.Classifier;

public class DecisionTreeClassifier extends Classifier implements HasTree {

	public DecisionTreeClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public TreeModel encodeModel(List<DataField> dataFields){
		DataField dataField = dataFields.get(0);

		Output output = new Output(PMMLUtil.createProbabilityFields(dataField));

		TreeModel treeModel = TreeModelUtil.encodeTreeModel(this, MiningFunctionType.CLASSIFICATION, dataFields, true)
			.setOutput(output);

		return treeModel;
	}

	@Override
	public Tree getTree(){
		return (Tree)get("tree_");
	}
}