/*
 * Copyright (c) 2026 Villu Ruusmann
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
package causalml.meta;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.Model;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Label;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;

abstract
public class BaseRLearner<E extends Estimator> extends BaseLearner<E> {

	public BaseRLearner(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<String> treatmentGroups = getTreatmentGroups();
		Map<String, E> modelTau = getModelTau();

		Label label = schema.getLabel();

		List<ContinuousLabel> continuousLabels = ScalarLabelUtil.toScalarLabels(ContinuousLabel.class, label);

		ClassDictUtil.checkSize(continuousLabels, treatmentGroups, modelTau.entrySet());

		List<Model> models = new ArrayList<>();

		for(int i = 0; i < continuousLabels.size(); i++){
			ContinuousLabel continuousLabel = continuousLabels.get(i);
			String treatmentGroup = treatmentGroups.get(i);

			E estimator = modelTau.get(treatmentGroup);

			Schema segmentSchema = schema.toRelabeledSchema(continuousLabel);

			Model model = estimator.encode(segmentSchema);

			models.add(model);
		}

		return BaseLearnerUtil.encodeModel(models);
	}

	public Map<String, E> getModelTau(){
		return getModels("models_tau");
	}
}