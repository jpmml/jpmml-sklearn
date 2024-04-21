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
package sktree.ensemble.forest;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.HasEstimatorEnsemble;
import sklearn.Regressor;
import sktree.tree.ObliqueDecisionTreeRegressor;

public class ObliqueRandomForestRegressor extends Regressor implements HasEstimatorEnsemble<ObliqueDecisionTreeRegressor> {

	public ObliqueRandomForestRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<ObliqueDecisionTreeRegressor> estimators = getEstimators();

		List<TreeModel> treeModels = new ArrayList<>();

		Schema segmentSchema = schema.toAnonymousSchema();

		for(int i = 0; i < estimators.size(); i++){
			ObliqueDecisionTreeRegressor estimator = estimators.get(i);

			TreeModel treeModel = (TreeModel)estimator.encode((i + 1), segmentSchema);

			treeModels.add(treeModel);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.AVERAGE, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels));

		return miningModel;
	}

	@Override
	public List<ObliqueDecisionTreeRegressor> getEstimators(){
		return getList("estimators_", ObliqueDecisionTreeRegressor.class);
	}
}