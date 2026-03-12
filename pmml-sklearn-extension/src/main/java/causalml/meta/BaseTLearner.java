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
import sklearn.tree.HasTreeOptions;
import sklearn.tree.TreeUtil;

abstract
public class BaseTLearner<E extends Estimator> extends BaseLearner<E> implements HasTreeOptions {

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

			Schema segmentSchema = schema.toRelabeledSchema(continuousLabel);

			Model controlModel = encodeEstimator(Role.CONTROL, controlEstimator, segmentSchema);
			Model treatmentModel = encodeEstimator(Role.TREATMENT, treatmentEstimator, segmentSchema);

			Model binaryModel = encodeBinaryModel(controlModel, treatmentModel, segmentSchema);

			binaryModels.add(binaryModel);
		}

		return BaseLearnerUtil.encodeModel(binaryModels);
	}

	@Override
	public Schema configureSchema(Schema schema){

		if(hasTreeOptions()){
			return TreeUtil.configureSchema(this, schema);
		}

		return super.configureSchema(schema);
	}

	@Override
	public Model configureModel(Model model){

		if(hasTreeOptions()){
			return TreeUtil.configureModel(this, model);
		}

		return super.configureModel(model);
	}

	protected MiningModel encodeBinaryModel(Model controlModel, Model treatmentModel, Schema schema){
		return BaseLearnerUtil.encodeBinaryRegressor(controlModel, treatmentModel, schema);
	}

	protected boolean hasTreeOptions(){
		E modelC = getModelC();
		E modelT = getModelT();

		return ((modelC instanceof HasTreeOptions) && (modelT instanceof HasTreeOptions));
	}

	public E getModelC(){
		return getModel("model_c");
	}

	public E getModelT(){
		return getModel("model_t");
	}

	public Map<String, E> getModelsC(){
		return getModels("models_c");
	}

	public Map<String, E> getModelsT(){
		return getModels("models_t");
	}
}