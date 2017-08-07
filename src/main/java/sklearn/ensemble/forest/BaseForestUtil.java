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
package sklearn.ensemble.forest;

import java.util.List;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Estimator;
import sklearn.tree.HasTree;
import sklearn.tree.TreeModelUtil;

public class BaseForestUtil {

	private BaseForestUtil(){
	}

	static
	public <E extends Estimator & HasTree> MiningModel encodeBaseForest(List<E> estimators, Segmentation.MultipleModelMethod multipleModelMethod, MiningFunction miningFunction, Schema schema){
		List<TreeModel> treeModels = TreeModelUtil.encodeTreeModelSegmentation(estimators, miningFunction, schema);

		MiningModel miningModel = new MiningModel(miningFunction, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(multipleModelMethod, treeModels));

		return miningModel;
	}
}