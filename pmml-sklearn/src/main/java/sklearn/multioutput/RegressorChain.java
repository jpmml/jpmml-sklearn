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
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;
import sklearn.Regressor;

public class RegressorChain extends Regressor {

	public RegressorChain(String module, String name){
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
		List<Integer> order = getOrder();

		ClassDictUtil.checkSize(estimators, order);

		PMMLEncoder encoder = schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<Model> models = new ArrayList<>();

		MultiLabel multiLabel = (MultiLabel)label;

		List<Feature> augmentedFeatures = new ArrayList<>(features);

		for(int i = 0; i < estimators.size(); i++){
			Regressor estimator = estimators.get(i);

			if(order.get(i) != i){
				throw new IllegalArgumentException();
			}

			ContinuousLabel continuousLabel = (ContinuousLabel)multiLabel.getLabel(i);

			Schema segmentSchema = new Schema(encoder, continuousLabel, augmentedFeatures);

			Model model = estimator.encodeModel(segmentSchema);

			models.add(model);

			OutputField predictedOutputField = ModelUtil.createPredictedField(FieldNameUtil.create(Estimator.FIELD_PREDICT, continuousLabel.getName()), OpType.CONTINUOUS, continuousLabel.getDataType())
				.setFinalResult(false);

			DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

			augmentedFeatures.add(new ContinuousFeature(encoder, predictedField));
		}

		return MiningModelUtil.createMultiModelChain(models, Segmentation.MissingPredictionTreatment.CONTINUE);
	}

	public List<? extends Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}

	public List<Integer> getOrder(){
		return getIntegerArray("order_");
	}
}