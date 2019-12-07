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
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasEstimatorEnsemble;
import sklearn.Regressor;

public class StackingRegressor extends Regressor implements HasEstimatorEnsemble<Regressor> {

	public StackingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();
		List<String> stackMethods = getStackMethod();
		Regressor finalEstimator = getFinalEstimator();
		Boolean passthrough = getPassthrough();

		ClassDictUtil.checkSize(estimators, stackMethods);

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Set<PMMLEncoder> encoders = features.stream()
			.map(feature -> feature.getEncoder())
			.collect(Collectors.toSet());

		SkLearnEncoder encoder = (SkLearnEncoder)Iterables.getOnlyElement(encoders);

		List<Feature> stackFeatures = new ArrayList<>();

		List<Model> models = new ArrayList<>();

		for(int i = 0; i < estimators.size(); i++){
			Regressor estimator = estimators.get(i);
			String stackMethod = stackMethods.get(i);

			if(!("predict").equals(stackMethod)){
				throw new IllegalArgumentException(stackMethod);
			}

			Model model = estimator.encodeModel(schema);

			OutputField predictedOutputField = ModelUtil.createPredictedField(FieldName.create(stackMethod + "(" + i + ")"), OpType.CONTINUOUS, label.getDataType());

			DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

			stackFeatures.add(new ContinuousFeature(encoder, predictedField));

			models.add(model);
		}

		if(passthrough){
			stackFeatures.addAll(features);
		}

		{
			Schema finalSchema = new Schema(label, stackFeatures);

			Model finalModel = finalEstimator.encodeModel(finalSchema);

			models.add(finalModel);
		}

		return MiningModelUtil.createModelChain(models);
	}

	@Override
	public List<? extends Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}

	public Regressor getFinalEstimator(){
		return get("final_estimator_", Regressor.class);
	}

	public Boolean getPassthrough(){
		return getBoolean("passthrough");
	}

	public List<String> getStackMethod(){
		return getList("stack_method_", String.class);
	}
}