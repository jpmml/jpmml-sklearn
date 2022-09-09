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

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.sklearn.SkLearnEncoder;
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

		List<Feature> result = EstimatorUtil.export(estimator, predictFunc, schema, model, encoder);

		Output output = model.getOutput();
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
}