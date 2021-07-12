/*
 * Copyright (c) 2021 Villu Ruusmann
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
package sklego.meta;

import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasDecisionFunctionField;
import sklearn.HasEstimator;
import sklearn.Transformer;

public class EstimatorTransformer extends Transformer implements HasEstimator<Estimator> {

	public EstimatorTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();
		String predictFunc = getPredictFunc();

		FieldName inputName;

		switch(predictFunc){
			case "predict":
				{
					inputName = null;
				}
				break;
			case "decision_function":
				{
					if(estimator instanceof HasDecisionFunctionField){
						HasDecisionFunctionField hasDecisionFunctionField = (HasDecisionFunctionField)estimator;

						inputName = hasDecisionFunctionField.getDecisionFunctionField();
					} else

					{
						throw new IllegalArgumentException();
					}
				}
				break;
			default:
				throw new IllegalArgumentException(predictFunc);
		}

		Schema schema = createSchema(estimator, features, encoder);

		Model model = estimator.encode(schema);

		Map<FieldName, DerivedOutputField> derivedOutputFields = new LinkedHashMap<>();

		Output output = model.getOutput();
		if(output != null && output.hasOutputFields()){
			List<OutputField> outputFields = output.getOutputFields();

			for(Iterator<OutputField> it = outputFields.iterator(); it.hasNext(); ){
				OutputField outputField = it.next();

				ResultFeature resultFeature = outputField.getResultFeature();
				switch(resultFeature){
					case PREDICTED_VALUE:
					case TRANSFORMED_VALUE:
						{
							DerivedOutputField derivedOutputField = encoder.createDerivedField(model, outputField, true);

							derivedOutputFields.put(derivedOutputField.getName(), derivedOutputField);
						}
						break;
					default:
						break;
				}

				it.remove();
			}
		}

		encoder.addTransformer(model);

		if(inputName == null){

			if(estimator.isSupervised()){

				if(!derivedOutputFields.isEmpty()){
					throw new IllegalArgumentException();
				}

				Label label = schema.getLabel();

				FieldName name = createFieldName(Estimator.FIELD_PREDICT);

				MiningFunction miningFunction = estimator.getMiningFunction();
				switch(miningFunction){
					case CLASSIFICATION:
						{
							CategoricalLabel categoricalLabel = (CategoricalLabel)label;

							OutputField predictedOutputField = ModelUtil.createPredictedField(name, OpType.CATEGORICAL, categoricalLabel.getDataType());

							DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

							return Collections.singletonList(new CategoricalFeature(encoder, predictedField, categoricalLabel.getValues()));
						}
					case REGRESSION:
						{
							ContinuousLabel continuousLabel = (ContinuousLabel)label;

							OutputField predictedOutputField = ModelUtil.createPredictedField(name, OpType.CONTINUOUS, continuousLabel.getDataType());

							DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

							return Collections.singletonList(new ContinuousFeature(encoder, predictedField));
						}
					default:
						throw new IllegalArgumentException();
				}
			} else

			{
				if(derivedOutputFields.isEmpty()){
					throw new IllegalArgumentException();
				}

				inputName = Iterables.getLast(derivedOutputFields.keySet());
			}
		}

		DerivedOutputField inputField = derivedOutputFields.get(inputName);
		if(inputField == null){
			throw new IllegalArgumentException();
		}

		OpType opType = inputField.getOpType();
		switch(opType){
			case CATEGORICAL:
				{
					OutputField outputField = inputField.getOutputField();

					if(!outputField.hasValues()){
						throw new IllegalArgumentException();
					}

					List<?> values = PMMLUtil.getValues(outputField);

					return Collections.singletonList(new CategoricalFeature(encoder, inputField, values));
				}
			case CONTINUOUS:
				{
					return Collections.singletonList(new ContinuousFeature(encoder, inputField));
				}
			default:
				throw new IllegalArgumentException();
		}
	}

	@Override
	public Estimator getEstimator(){
		return get("estimator_", Estimator.class);
	}

	public String getPredictFunc(){
		return getString("predict_func");
	}

	static
	private Schema createSchema(Estimator estimator, List<Feature> features, SkLearnEncoder encoder){
		Label label = null;

		if(estimator.isSupervised()){
			MiningFunction miningFunction = estimator.getMiningFunction();

			switch(miningFunction){
				case CLASSIFICATION:
					{
						List<?> categories = ClassifierUtil.getClasses(estimator);

						DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

						label = new CategoricalLabel(null, dataType, categories);
					}
					break;
				case REGRESSION:
					{
						label = new ContinuousLabel(null, DataType.DOUBLE);
					}
					break;
				default:
					throw new IllegalArgumentException();
			}
		}

		return new Schema(encoder, label, features);
	}
}