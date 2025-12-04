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
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.HasNativeConfiguration;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.HasSkLearnOptions;
import org.jpmml.sklearn.SkLearnEncoder;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public MiningFunction getMiningFunction(List<? extends Estimator> estimators){
		Set<MiningFunction> miningFunctions = estimators.stream()
			.map(estimator -> estimator.getMiningFunction())
			.collect(Collectors.toSet());

		if(miningFunctions.size() == 1){
			return Iterables.getOnlyElement(miningFunctions);
		}

		return MiningFunction.MIXED;
	}

	static
	public List<?> getClasses(Estimator estimator){

		if(estimator instanceof HasClasses){
			HasClasses hasClasses = (HasClasses)estimator;

			return hasClasses.getClasses();
		} else

		{
			throw new EstimatorException(estimator, HasClasses.class);
		}
	}

	static
	public boolean hasProbabilityDistribution(Estimator estimator){

		if(estimator instanceof HasClasses){
			HasClasses hasClasses = (HasClasses)estimator;

			return hasClasses.hasProbabilityDistribution();
		} else

		{
			throw new EstimatorException(estimator, HasClasses.class);
		}
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

						return encoder.export(model, hasMultiApplyField.getMultiApplyFields());
					} else

					{
						throw new EstimatorException(estimator, HasApplyField.class, HasMultiApplyField.class);
					}
				}
			case SkLearnMethods.DECISION_FUNCTION:
				{
					if(estimator instanceof HasDecisionFunctionField){
						HasDecisionFunctionField hasDecisionFunctionField = (HasDecisionFunctionField)estimator;

						return encoder.export(model, hasDecisionFunctionField.getDecisionFunctionField());
					} else

					{
						throw new EstimatorException(estimator, HasDecisionFunctionField.class);
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

						Feature feature = encoder.exportPrediction(model, scalarLabel);

						return Collections.singletonList(feature);
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

							return encoder.export(model, outputField.requireName());
						} else

						{
							throw new IllegalArgumentException();
						}
					}
				}
			case SkLearnMethods.PREDICT_PROBA:
				{
					if(estimator instanceof HasClasses){
						HasClasses hasClasses = (HasClasses)estimator;

						CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

						List<OutputField> probabilityOutputFields = estimator.createPredictProbaFields(DataType.DOUBLE, categoricalLabel);

						List<String> names = probabilityOutputFields.stream()
							.map(outputField -> outputField.requireName())
							.collect(Collectors.toList());

						return encoder.export(model, names);
					} else

					{
						throw new EstimatorException(estimator, HasClasses.class);
					}
				}
			default:
				throw new IllegalArgumentException(predictFunc);
		}
	}

	static
	public <E extends Estimator & HasFeatureNamesIn & HasSkLearnOptions> PMML encodePMML(E estimator){
		SkLearnEncoder encoder = new SkLearnEncoder();

		if(estimator.isSupervised()){
			List<String> targetFields = EncodableUtil.generateOutputNames(estimator);

			encoder.initLabel(estimator, targetFields);
		}

		List<String> activeFields = EncodableUtil.getOrGenerateFeatureNames(estimator);

		encoder.initFeatures(estimator, activeFields);

		Schema schema = encoder.createSchema();

		Model model = encodeNativeLike(estimator, schema);

		encoder.setModel(model);

		return encoder.encodePMML(model);
	}

	static
	public Model encodeNativeLike(Estimator estimator, Schema schema){

		if(estimator instanceof HasNativeConfiguration){
			HasNativeConfiguration hasNativeConfiguration = (HasNativeConfiguration)estimator;

			Map<String, ?> prevPmmlOptions = estimator.getPMMLOptions();

			try {
				estimator.setPMMLOptions(hasNativeConfiguration.getNativeConfiguration());

				return estimator.encode(schema);
			} finally {
				estimator.setPMMLOptions(prevPmmlOptions);
			}
		} else

		{
			return estimator.encode(schema);
		}
	}

	static
	public Output getFinalOutput(Model model){
		model = MiningModelUtil.getFinalModel(model);

		return model.getOutput();
	}
}
