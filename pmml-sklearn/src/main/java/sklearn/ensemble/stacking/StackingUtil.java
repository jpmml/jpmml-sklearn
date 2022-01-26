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
package sklearn.ensemble.stacking;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;

public class StackingUtil {

	private StackingUtil(){
	}

	static
	public <E extends Estimator> MiningModel encodeStacking(List<? extends E> estimators, List<String> stackMethods, PredictFunction predictFunction, E finalEstimator, boolean passthrough, Schema schema){
		ClassDictUtil.checkSize(estimators, stackMethods);

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<Feature> stackFeatures = new ArrayList<>();

		List<Model> models = new ArrayList<>();

		for(int i = 0; i < estimators.size(); i++){
			E estimator = estimators.get(i);
			String stackMethod = stackMethods.get(i);

			Model model = estimator.encode(schema);

			List<Feature> predictFeatures = predictFunction.apply(i, model, stackMethod, encoder);
			if(predictFeatures != null && predictFeatures.size() > 0){
				stackFeatures.addAll(predictFeatures);
			}

			models.add(model);
		}

		if(passthrough){
			stackFeatures.addAll(features);
		}

		{
			Schema stackSchema = new Schema(encoder, label, stackFeatures);

			Model finalModel = finalEstimator.encode(stackSchema);

			models.add(finalModel);
		}

		return MiningModelUtil.createModelChain(models);
	}

	static
	public interface PredictFunction {

		List<Feature> apply(int index, Model model, String stackMethod, SkLearnEncoder encoder);
	}
}