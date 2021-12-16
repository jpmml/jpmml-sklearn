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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
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
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasApplyField;
import sklearn.HasDecisionFunctionField;
import sklearn.HasEstimator;
import sklearn.HasMultiApplyField;
import sklearn.HasPredictField;
import sklearn.SkLearnMethods;
import sklearn.Transformer;
import sklearn.tree.HasTreeOptions;

public class EstimatorTransformer extends Transformer implements HasEstimator<Estimator> {

	public EstimatorTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();
		String predictFunc = getPredictFunc();

		List<String> inputNames;

		switch(predictFunc){
			case SkLearnMethods.APPLY:
				{
					if(estimator instanceof HasTreeOptions){
						HasTreeOptions hasTreeOptions = (HasTreeOptions)estimator;

						// XXX
						estimator.putOption(HasTreeOptions.OPTION_WINNER_ID, Boolean.TRUE);
					} // End if

					if(estimator instanceof HasApplyField){
						HasApplyField hasApplyField = (HasApplyField)estimator;

						inputNames = Collections.singletonList(hasApplyField.getApplyField());
					} else

					if(estimator instanceof HasMultiApplyField){
						HasMultiApplyField hasMultiApplyField = (HasMultiApplyField)estimator;

						inputNames = hasMultiApplyField.getApplyFields();
					} else

					{
						throw new IllegalArgumentException();
					}
				}
				break;
			case SkLearnMethods.DECISION_FUNCTION:
				{
					if(estimator instanceof HasDecisionFunctionField){
						HasDecisionFunctionField hasDecisionFunctionField = (HasDecisionFunctionField)estimator;

						inputNames = Collections.singletonList(hasDecisionFunctionField.getDecisionFunctionField());
					} else

					{
						throw new IllegalArgumentException();
					}
				}
				break;
			case SkLearnMethods.PREDICT:
				{
					if(estimator instanceof HasPredictField){
						HasPredictField hasPredictField = (HasPredictField)estimator;

						inputNames = Collections.singletonList(hasPredictField.getPredictField());
					} else

					{
						inputNames = null;
					}
				}
				break;
			default:
				throw new IllegalArgumentException(predictFunc);
		}

		Schema schema = createSchema(estimator, features, encoder);

		Model model = estimator.encode(schema);

		Map<String, DerivedOutputField> derivedOutputFields = new LinkedHashMap<>();

		Output output = model.getOutput();
		if(output != null && output.hasOutputFields()){
			List<OutputField> outputFields = output.getOutputFields();

			for(Iterator<OutputField> it = outputFields.iterator(); it.hasNext(); ){
				OutputField outputField = it.next();

				ResultFeature resultFeature = outputField.getResultFeature();
				switch(resultFeature){
					case PREDICTED_VALUE:
					case TRANSFORMED_VALUE:
					case DECISION:
					case ENTITY_ID:
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

		if(inputNames == null){

			if(estimator.isSupervised()){

				if(!derivedOutputFields.isEmpty()){
					throw new IllegalArgumentException();
				}

				Label label = schema.getLabel();

				String name = createFieldName(Estimator.FIELD_PREDICT);

				MiningFunction miningFunction = estimator.getMiningFunction();
				switch(miningFunction){
					case CLASSIFICATION:
						{
							CategoricalLabel categoricalLabel = (CategoricalLabel)label;

							List<?> categories = categoricalLabel.getValues();

							OutputField predictedOutputField = ModelUtil.createPredictedField(name, OpType.CATEGORICAL, categoricalLabel.getDataType());

							DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

							return Collections.singletonList(new CategoricalFeature(encoder, predictedField, categories));
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

				inputNames = Collections.singletonList(Iterables.getLast(derivedOutputFields.keySet()));
			}
		}

		List<Feature> result = new ArrayList<>();

		for(String inputName : inputNames){
			DerivedOutputField inputField = derivedOutputFields.get(inputName);
			if(inputField == null){
				throw new IllegalArgumentException();
			}

			Feature feature;

			OpType opType = inputField.getOpType();
			switch(opType){
				case CATEGORICAL:
					feature = new CategoricalFeature(encoder, inputField.getOutputField());
					break;
				case CONTINUOUS:
					feature = new ContinuousFeature(encoder, inputField);
					break;
				default:
					throw new IllegalArgumentException();
			}

			result.add(feature);
		}

		return result;
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