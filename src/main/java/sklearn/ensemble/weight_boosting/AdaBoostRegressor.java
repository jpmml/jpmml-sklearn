/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.ensemble.weight_boosting;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Estimator;
import sklearn.Regressor;
import sklearn.ensemble.EnsembleRegressor;

public class AdaBoostRegressor extends EnsembleRegressor {

	public AdaBoostRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<? extends Regressor> estimators = getEstimators();

		Estimator estimator = estimators.get(0);

		return estimator.getNumberOfFeatures();
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();
		List<? extends Number> estimatorWeights = getEstimatorWeights();

		Schema segmentSchema = schema.toAnonymousSchema();

		List<Model> models = new ArrayList<>();

		for(Regressor estimator : estimators){
			Model model = estimator.encodeModel(segmentSchema);

			models.add(model);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.WEIGHTED_MEDIAN, models, estimatorWeights));

		return miningModel;
	}

	public List<? extends Number> getEstimatorWeights(){
		return (List)ClassDictUtil.getArray(this, "estimator_weights_");
	}
}