/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.ensemble.stacking;

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasEstimatorEnsemble;
import sklearn.Regressor;

public class StackingRegressor extends Regressor implements HasEstimatorEnsemble<Regressor> {

	public StackingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();
		Regressor finalEstimator = getFinalEstimator();
		Boolean passthrough = getPassthrough();
		List<String> stackMethod = getStackMethod();

		ContinuousLabel continuousLabel = (ContinuousLabel)schema.getLabel();

		StackingUtil.PredictFunction predictFunction = new StackingUtil.PredictFunction(){

			@Override
			public List<Feature> apply(int index, Model model, String stackMethod, SkLearnEncoder encoder){

				if(!("predict").equals(stackMethod)){
					throw new IllegalArgumentException(stackMethod);
				}

				OutputField predictedOutputField = ModelUtil.createPredictedField(FieldNameUtil.create(stackMethod, index), OpType.CONTINUOUS, continuousLabel.getDataType());

				DerivedOutputField predictedField = encoder.createDerivedField(model, predictedOutputField, false);

				return Collections.singletonList(new ContinuousFeature(encoder, predictedField));
			}
		};

		return StackingUtil.encodeStacking(estimators, stackMethod, predictFunction, finalEstimator, passthrough, schema);
	}

	@Override
	public List<? extends Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}

	public Regressor getFinalEstimator(){
		return get("final_estimator_", Regressor.class);
	}

	public Boolean getPassthrough(){
		return getBoolean("passthrough");
	}

	public List<String> getStackMethod(){
		return getList("stack_method_", String.class);
	}
}