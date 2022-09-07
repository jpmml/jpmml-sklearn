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
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

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
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasApplyField;
import sklearn.HasClasses;
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

		switch(predictFunc){
			case SkLearnMethods.APPLY:
			case SkLearnMethods.DECISION_FUNCTION:
			case SkLearnMethods.PREDICT:
			case SkLearnMethods.PREDICT_PROBA:
				break;
			default:
				throw new IllegalArgumentException(predictFunc);
		}

		Schema schema = createSchema(estimator, features, encoder);

		switch(predictFunc){
			case SkLearnMethods.APPLY:
				{
					if(estimator instanceof HasTreeOptions){
						HasTreeOptions hasTreeOptions = (HasTreeOptions)estimator;

						// XXX
						estimator.putOption(HasTreeOptions.OPTION_WINNER_ID, Boolean.TRUE);
					}
				}
				break;
			default:
				break;
		}

		Model model = estimator.encode(schema);

		encoder.addTransformer(model);

		Output output = model.getOutput();

		List<Feature> result;

		switch(predictFunc){
			case SkLearnMethods.APPLY:
				{
					if(estimator instanceof HasApplyField){
						HasApplyField hasApplyField = (HasApplyField)estimator;

						result = export(model, hasApplyField.getApplyField(), encoder);
					} else

					if(estimator instanceof HasMultiApplyField){
						HasMultiApplyField hasMultiApplyField = (HasMultiApplyField)estimator;

						result = export(model, hasMultiApplyField.getApplyFields(), encoder);
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

						result = export(model, hasDecisionFunctionField.getDecisionFunctionField(), encoder);
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

						result = export(model, hasPredictField.getPredictField(), encoder);
					} else

					{
						if(estimator.isSupervised()){
							ScalarLabel scalarLabel = (ScalarLabel)schema.getLabel();

							MiningFunction miningFunction = estimator.getMiningFunction();
							switch(miningFunction){
								case CLASSIFICATION:
								case REGRESSION:
									{
										Feature feature = encoder.exportPrediction(model, scalarLabel);

										result = Collections.singletonList(feature);
									}
									break;
								default:
									throw new IllegalArgumentException();
							}
						} else

						{
							if(output != null && output.hasOutputFields()){
								List<OutputField> outputFields = output.getOutputFields();

								List<OutputField> predictionOutputFields = outputFields.stream()
									.filter(outputField -> isPrediction(outputField))
									.collect(Collectors.toList());

								if(predictionOutputFields.isEmpty()){
									throw new IllegalArgumentException();
								}

								OutputField outputField = Iterables.getLast(predictionOutputFields);

								result = export(model, outputField.getName(), encoder);
							} else

							{
								throw new IllegalArgumentException();
							}
						}
					}
				}
				break;
			case SkLearnMethods.PREDICT_PROBA:
				{
					if(estimator instanceof HasClasses){
						List<?> categories = EstimatorUtil.getClasses(estimator);

						List<String> names = categories.stream()
							.map(category -> FieldNameUtil.create(Classifier.FIELD_PROBABILITY, category))
							.collect(Collectors.toList());

						result = export(model, names, encoder);
					} else

					{
						throw new IllegalArgumentException();
					}
				}
				break;
			default:
				throw new IllegalArgumentException(predictFunc);
		}

		if(output != null && output.hasOutputFields()){
			List<OutputField> outputFields = output.getOutputFields();

			outputFields.clear();
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
						List<?> categories = EstimatorUtil.getClasses(estimator);

						DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

						label = new CategoricalLabel(dataType, categories);
					}
					break;
				case REGRESSION:
					{
						label = new ContinuousLabel(DataType.DOUBLE);
					}
					break;
				default:
					throw new IllegalArgumentException();
			}
		}

		return new Schema(encoder, label, features);
	}

	static
	private List<Feature> export(Model model, String name, SkLearnEncoder encoder){
		return export(model, Collections.singletonList(name), encoder);
	}

	static
	private List<Feature> export(Model model, List<String> names, SkLearnEncoder encoder){
		Output output = model.getOutput();

		List<OutputField> outputFields = output.getOutputFields();

		List<Feature> result = new ArrayList<>();

		for(String name : names){
			DerivedOutputField derivedOutputField = null;

			List<OutputField> nameOutputFields = selectOutputFields(name, outputFields);
			for(OutputField nameOutputField : nameOutputFields){
				derivedOutputField = encoder.createDerivedField(model, nameOutputField, true);
			}

			Feature feature = toFeature(derivedOutputField.getOutputField(), encoder);

			result.add(feature);

			outputFields.removeAll(nameOutputFields);
		}

		return result;
	}

	static
	private List<OutputField> selectOutputFields(String name, List<OutputField> outputFields){
		List<OutputField> result = new ArrayList<>();

		for(OutputField outputField : outputFields){
			boolean prediction = isPrediction(outputField);

			if(prediction){
				result.add(outputField);
			} // End if

			if(!Objects.equals(name, outputField.requireName())){
				continue;
			} // End if

			if(prediction){
				return result;
			} else

			{
				return Collections.singletonList(outputField);
			}
		}

		throw new IllegalArgumentException(name);
	}

	static
	private boolean isPrediction(OutputField outputField){
		ResultFeature resultFeature = outputField.getResultFeature();

		switch(resultFeature){
			case PREDICTED_VALUE:
			case TRANSFORMED_VALUE:
			case DECISION:
				return true;
			default:
				return false;
		}
	}

	static
	private Feature toFeature(OutputField outputField, SkLearnEncoder encoder){
		OpType opType = outputField.getOpType();

		switch(opType){
			case CATEGORICAL:
				return new CategoricalFeature(encoder, outputField);
			case CONTINUOUS:
				return new ContinuousFeature(encoder, outputField);
			default:
				throw new IllegalArgumentException();
		}
	}
}