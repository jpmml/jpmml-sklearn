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
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.HasEstimatorEnsemble;

public class StackingClassifier extends Classifier implements HasEstimatorEnsemble<Classifier> {

	public StackingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<? extends Classifier> estimators = getEstimators();
		Classifier finalEstimator = getFinalEstimator();
		Boolean passthrough = getPassthrough();
		List<String> stackMethod = getStackMethod();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		List<?> values;

		if(categoricalLabel.size() == 2){
			values = Collections.singletonList(categoricalLabel.getValue(1));
		} else

		{
			values = categoricalLabel.getValues();
		}

		StackingUtil.PredictFunction predictFunction = new StackingUtil.PredictFunction(){

			@Override
			public List<Feature> apply(int index, Model model, String stackMethod, SkLearnEncoder encoder){

				if(!("predict_proba").equals(stackMethod)){
					throw new IllegalArgumentException(stackMethod);
				}

				Model finalModel = MiningModelUtil.getFinalModel(model);

				Output finalOutput = finalModel.getOutput();
				if(finalOutput != null && finalOutput.hasOutputFields()){
					List<OutputField> outputFields = finalOutput.getOutputFields();

					outputFields.stream()
						.filter(outputField -> (ResultFeature.PROBABILITY).equals(outputField.getResultFeature()))
						.forEach(outputField -> {
							FieldName name = outputField.getName();

							String stringValue = name.getValue();

							if(stringValue.startsWith("probability(")){
								FieldName indexedName = FieldNameUtil.create("probability", index, stringValue.substring("probability(".length()));

								stringValue = indexedName.getValue();
							}

							outputField.setName(FieldName.create(stringValue));
						});
				}

				List<Feature> result = new ArrayList<>();

				for(Object value : values){
					OutputField probabilityOutputField = ModelUtil.createProbabilityField(FieldNameUtil.create(stackMethod, index, value), DataType.DOUBLE, value);

					DerivedOutputField predictedField = encoder.createDerivedField(model, probabilityOutputField, false);

					result.add(new ContinuousFeature(encoder, predictedField));
				}

				return result;
			}
		};

		return StackingUtil.encodeStacking(estimators, stackMethod, predictFunction, finalEstimator, passthrough, schema);
	}

	@Override
	public List<? extends Classifier> getEstimators(){
		return getList("estimators_", Classifier.class);
	}

	public Classifier getFinalEstimator(){
		return get("final_estimator_", Classifier.class);
	}

	public Boolean getPassthrough(){
		return getBoolean("passthrough");
	}

	public List<String> getStackMethod(){
		return getList("stack_method_", String.class);
	}
}