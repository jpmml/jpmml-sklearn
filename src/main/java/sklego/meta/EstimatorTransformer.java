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
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
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

		if(!estimator.isSupervised()){
			throw new IllegalArgumentException();
		}

		switch(predictFunc){
			case "predict":
				break;
			default:
				throw new IllegalArgumentException(predictFunc);
		}

		MiningFunction miningFunction = estimator.getMiningFunction();

		Label label;

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

		Schema schema = new Schema(encoder, label, features);

		Model model = estimator.encode(schema);

		Output output = model.getOutput();
		if(output != null && output.hasOutputFields()){
			List<OutputField> outputFields = output.getOutputFields();

			outputFields.clear();
		}

		encoder.addTransformer(model);

		FieldName name = createFieldName("estimator");

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
	}

	@Override
	public Estimator getEstimator(){
		return get("estimator_", Estimator.class);
	}

	public String getPredictFunc(){
		return getString("predict_func");
	}
}