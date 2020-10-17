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
import org.dmg.pmml.OutputField;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasEstimator;
import sklearn.Transformer;

public class StackingEstimator extends Transformer implements HasEstimator<Estimator> {

	public StackingEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		Estimator estimator = getEstimator();

		return estimator.getNumberOfFeatures();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();

		Label label;

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

		Schema schema = new Schema(encoder, label, features);

		Model model = estimator.encode(schema);

		encoder.addTransformer(model);

		FieldName name = createFieldName("stack", features);

		List<Feature> result = new ArrayList<>();

		switch(miningFunction){
			case CLASSIFICATION:
				{
					Classifier classifier = (Classifier)estimator;

					List<?> categories = ClassifierUtil.getClasses(estimator);

					if(classifier.hasProbabilityDistribution()){

						for(Object category : categories){
							OutputField probabilityOutputField = ModelUtil.createProbabilityField(FieldNameUtil.create("probability", name, category), DataType.DOUBLE, category)
								.setFinalResult(false);

							DerivedOutputField probabilityField = encoder.createDerivedField(model, probabilityOutputField, false);

							result.add(new ContinuousFeature(encoder, probabilityField));
						}
					}

					OutputField predictedOutputField = ModelUtil.createPredictedField(name, OpType.CATEGORICAL, label.getDataType());

					DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

					result.add(new CategoricalFeature(encoder, predictedField, categories));
				}
				break;
			case REGRESSION:
				{
					OutputField predictedOutputField = ModelUtil.createPredictedField(name, OpType.CONTINUOUS, label.getDataType());

					DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

					result.add(new ContinuousFeature(encoder, predictedField));
				}
				break;
			default:
				throw new IllegalArgumentException();
		}

		result.addAll(features);

		return result;
	}

	@Override
	public Estimator getEstimator(){
		return get("estimator", Estimator.class);
	}
}