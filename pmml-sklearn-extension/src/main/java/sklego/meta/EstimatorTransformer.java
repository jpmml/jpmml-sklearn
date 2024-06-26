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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Calibrator;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasEstimator;
import sklearn.SkLearnMethods;
import sklearn.Transformer;
import sklearn.tree.HasTreeOptions;

public class EstimatorTransformer extends Transformer implements HasEstimator<Estimator> {

	public EstimatorTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){

		if(!encoder.hasModel()){
			return encodePreProcessor(features, encoder);
		} else

		{
			return encodePostProcessor(features, encoder);
		}
	}

	private List<Feature> encodePreProcessor(List<Feature> features, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();
		String predictFunc = getPrePredictFunc();

		ScalarLabel scalarLabel = null;

		if(estimator.isSupervised()){
			scalarLabel = (ScalarLabel)estimator.encodeLabel(Collections.singletonList(null), encoder);
		}

		Schema schema = new Schema(encoder, scalarLabel, features);

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

		List<Feature> result = EstimatorUtil.export(estimator, predictFunc, schema, model, encoder);

		Output output = model.getOutput();
		if(output != null && output.hasOutputFields()){
			List<OutputField> outputFields = output.getOutputFields();

			outputFields.clear();
		}

		return result;
	}

	private List<Feature> encodePostProcessor(List<Feature> features, SkLearnEncoder encoder){
		Calibrator estimator = getEstimator(Calibrator.class);
		@SuppressWarnings("unused")
		String predictFunc = getPostPredictFunc();

		Model model = encoder.getModel();

		features = estimator.encodeFeatures(features, encoder);

		ContinuousFeature continuousFeature = (ContinuousFeature)Iterables.getOnlyElement(features);

		DerivedField derivedField = (DerivedField)continuousFeature.getField();

		String name = derivedField.requireName();

		encoder.removeDerivedField(name);

		OutputField outputField = new OutputField(name, derivedField.requireOpType(), derivedField.requireDataType())
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(derivedField.requireExpression());

		Output output = ModelUtil.ensureOutput(model);

		output.addOutputFields(outputField);

		return encoder.export(model, name);
	}

	@Override
	public Estimator getEstimator(){
		return getEstimator(Estimator.class);
	}

	public <E extends Estimator> E getEstimator(Class<? extends E> clazz){
		return get("estimator_", clazz);
	}

	public String getPredictFunc(){
		return getString("predict_func");
	}

	public String getPrePredictFunc(){
		return getEnum("predict_func", this::getString, Arrays.asList(SkLearnMethods.APPLY, SkLearnMethods.DECISION_FUNCTION, SkLearnMethods.PREDICT, SkLearnMethods.PREDICT_PROBA));
	}

	public String getPostPredictFunc(){
		return getEnum("predict_func", this::getString, Arrays.asList(SkLearnMethods.PREDICT));
	}
}