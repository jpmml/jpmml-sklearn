/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public List<?> getClasses(Estimator estimator){

		if(estimator instanceof HasClasses){
			HasClasses hasClasses = (HasClasses)estimator;

			return hasClasses.getClasses();
		}

		throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(estimator) + ") is not a classifier");
	}

	static
	public List<Feature> export(Estimator estimator, String predictFunc, Schema schema, Model model, SkLearnEncoder encoder){

		switch(predictFunc){
			case SkLearnMethods.APPLY:
				{
					if(estimator instanceof HasApplyField){
						HasApplyField hasApplyField = (HasApplyField)estimator;

						return encoder.export(model, hasApplyField.getApplyField());
					} else

					if(estimator instanceof HasMultiApplyField){
						HasMultiApplyField hasMultiApplyField = (HasMultiApplyField)estimator;

						return encoder.export(model, hasMultiApplyField.getApplyFields());
					} else

					{
						throw new IllegalArgumentException();
					}
				}
			case SkLearnMethods.DECISION_FUNCTION:
				{
					if(estimator instanceof HasDecisionFunctionField){
						HasDecisionFunctionField hasDecisionFunctionField = (HasDecisionFunctionField)estimator;

						return encoder.export(model, hasDecisionFunctionField.getDecisionFunctionField());
					} else

					{
						throw new IllegalArgumentException();
					}
				}
			case SkLearnMethods.PREDICT:
				{
					if(estimator instanceof HasPredictField){
						HasPredictField hasPredictField = (HasPredictField)estimator;

						return encoder.export(model, hasPredictField.getPredictField());
					} // End if

					if(estimator.isSupervised()){
						ScalarLabel scalarLabel = (ScalarLabel)schema.getLabel();

						MiningFunction miningFunction = estimator.getMiningFunction();
						switch(miningFunction){
							case CLASSIFICATION:
							case REGRESSION:
								{
									Feature feature = encoder.exportPrediction(model, scalarLabel);

									return Collections.singletonList(feature);
								}
							default:
								throw new IllegalArgumentException();
						}
					} else

					{
						Output output = model.getOutput();

						if(output != null && output.hasOutputFields()){
							List<OutputField> outputFields = output.getOutputFields();

							List<OutputField> predictionOutputFields = outputFields.stream()
								.filter(outputField -> SkLearnEncoder.isPrediction(outputField))
								.collect(Collectors.toList());

							if(predictionOutputFields.isEmpty()){
								throw new IllegalArgumentException();
							}

							OutputField outputField = Iterables.getLast(predictionOutputFields);

							return encoder.export(model, outputField.getName());
						} else

						{
							throw new IllegalArgumentException();
						}
					}
				}
			case SkLearnMethods.PREDICT_PROBA:
				{
					if(estimator instanceof HasClasses){
						List<?> categories = EstimatorUtil.getClasses(estimator);

						List<String> names = categories.stream()
							.map(category -> FieldNameUtil.create(Classifier.FIELD_PROBABILITY, category))
							.collect(Collectors.toList());

						return encoder.export(model, names);
					} else

					{
						throw new IllegalArgumentException();
					}
				}
			default:
				throw new IllegalArgumentException(predictFunc);
		}
	}
}
