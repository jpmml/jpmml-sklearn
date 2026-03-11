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
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Label;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;

abstract
public class BaseTLearner<E extends Estimator> extends BaseLearner<E> {

	public BaseTLearner(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<String> treatmentGroups = getTreatmentGroups();
		Map<String, E> controlModels = getModelsC();
		Map<String, E> treatmentModels = getModelsT();

		Label label = schema.getLabel();

		List<ContinuousLabel> continuousLabels = ScalarLabelUtil.toScalarLabels(ContinuousLabel.class, label);

		ClassDictUtil.checkSize(continuousLabels, treatmentGroups, treatmentModels.entrySet(), controlModels.entrySet());

		List<Model> binaryModels = new ArrayList<>();

		for(int i = 0; i < continuousLabels.size(); i++){
			ContinuousLabel continuousLabel = continuousLabels.get(i);
			String treatmentGroup = treatmentGroups.get(i);

			E controlEstimator = controlModels.get(treatmentGroup);
			E treatmentEstimator = treatmentModels.get(treatmentGroup);

			Schema binarySchema = schema.toRelabeledSchema(continuousLabel);

			Model controlModel = encodeEstimator(Role.CONTROL, controlEstimator, binarySchema);
			Model treatmentModel = encodeEstimator(Role.TREATMENT, treatmentEstimator, binarySchema);

			Model binaryModel = encodeBinaryModel(controlModel, treatmentModel, binarySchema);

			binaryModels.add(binaryModel);
		}

		return BaseLearnerUtil.encodeModel(binaryModels);
	}

	protected MiningModel encodeBinaryModel(Model controlModel, Model treatmentModel, Schema schema){
		return BaseLearnerUtil.encodeBinaryModel(controlModel, treatmentModel, schema);
	}

	public Map<String, E> getModelsC(){
		return getModels("models_c");
	}

	public Map<String, E> getModelsT(){
		return getModels("models_t");
	}
}