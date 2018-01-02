/*
 * Copyright (c) 2017 Villu Ruusmann
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
package tpot.builtins;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasEstimator;
import sklearn.Transformer;
import sklearn.TypeUtil;

public class StackingEstimator extends Transformer implements HasEstimator<Estimator> {

	public StackingEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();

		List<Feature> result = new ArrayList<>();

		FieldName name = FieldName.create("stack(" + features.size() + ")");

		Label label;

		Output output;

		MiningFunction miningFunction = estimator.getMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
				{
					Classifier classifier = (Classifier)estimator;

					List<?> classes = ClassifierUtil.getClasses(estimator);

					DataType dataType = TypeUtil.getDataType(classes, DataType.STRING);

					List<String> categories = ClassifierUtil.formatTargetCategories(classes);

					label = new CategoricalLabel(null, dataType, categories);

					output = ModelUtil.createPredictedOutput(name, OpType.CATEGORICAL, label.getDataType());

					if(classifier.hasProbabilityDistribution()){

						for(String category : categories){
							OutputField outputField = ModelUtil.createProbabilityField(DataType.DOUBLE, category)
								.setName(FieldName.create("probability(" + name.getValue() + ", " + category + ")"))
								.setFinalResult(false);

							output.addOutputFields(outputField);

							result.add(new ContinuousOutputFeature(encoder, output, outputField));
						}
					}

					result.add(new CategoricalOutputFeature(encoder, output, name, label.getDataType(), categories));
				}
				break;
			case REGRESSION:
				{
					label = new ContinuousLabel(null, DataType.DOUBLE);

					output = ModelUtil.createPredictedOutput(name, OpType.CONTINUOUS, label.getDataType());

					result.add(new ContinuousOutputFeature(encoder, output, name, label.getDataType()));
				}
				break;
			default:
				throw new IllegalArgumentException();
		}

		Schema schema = new Schema(label, features);

		Model model = estimator.encodeModel(schema)
			.setOutput(output);

		encoder.addTransformer(model);

		result.addAll(features);

		return result;
	}

	@Override
	public Estimator getEstimator(){
		return get("estimator", Estimator.class);
	}
}