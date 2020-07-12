/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.ensemble.voting;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.HasEstimatorEnsemble;
import sklearn.Regressor;

public class VotingRegressor extends Regressor implements HasEstimatorEnsemble<Regressor> {

	public VotingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();
		List<? extends Number> weights = getWeights();

		List<Model> models = new ArrayList<>();

		for(Regressor estimator : estimators){
			Model model = estimator.encode(schema);

			models.add(model);
		}

		Segmentation.MultipleModelMethod multipleModelMethod = (weights != null && weights.size() > 0 ? Segmentation.MultipleModelMethod.WEIGHTED_AVERAGE : Segmentation.MultipleModelMethod.AVERAGE);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(multipleModelMethod, models, weights));

		return miningModel;
	}

	@Override
	public List<? extends Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}

	public List<? extends Number> getWeights(){
		Object weights = getOptionalObject("weights");

		if((weights == null) || (weights instanceof List)){
			return (List)weights;
		}

		return getNumberArray("weights");
	}
}