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
package sklearn.multioutput;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.Model;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Estimator;
import sklearn.Regressor;

public class MultiOutputRegressor extends Regressor {

	public MultiOutputRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfOutputs(){
		List<? extends Regressor> estimators = getEstimators();

		return estimators.size();
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();

		if(estimators.size() == 1){
			Estimator estimator = estimators.get(0);

			return estimator.encodeModel(schema);
		} else

		if(estimators.size() >= 2){
			MultiLabel multiLabel = (MultiLabel)schema.getLabel();

			List<Model> models = new ArrayList<>();

			for(int i = 0; i < estimators.size(); i++){
				Regressor estimator = estimators.get(i);

				Schema segmentSchema = schema.toRelabeledSchema(multiLabel.getLabel(i));

				Model model = estimator.encodeModel(segmentSchema);

				models.add(model);
			}

			return MiningModelUtil.createMultiModelChain(models, Segmentation.MissingPredictionTreatment.CONTINUE);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<? extends Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}
}