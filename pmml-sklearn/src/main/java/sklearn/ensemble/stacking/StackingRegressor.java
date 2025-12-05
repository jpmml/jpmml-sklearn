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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;
import sklearn.Regressor;
import sklearn.SkLearnMethods;
import sklearn.SkLearnRegressor;
import sklearn.StepUtil;

public class StackingRegressor extends SkLearnRegressor implements HasEstimatorEnsemble<Regressor> {

	public StackingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		List<Regressor> estimators = getEstimators();

		return StepUtil.getNumberOfFeatures(estimators);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<Regressor> estimators = getEstimators();
		Regressor finalEstimator = getFinalEstimator();
		Boolean passthrough = getPassthrough();
		List<String> stackMethod = getStackMethod();

		ContinuousLabel continuousLabel = (ContinuousLabel)schema.getLabel();

		StackingUtil.PredictFunction predictFunction = new StackingUtil.PredictFunction(){

			@Override
			public List<Feature> apply(int index, Model model, String stackMethod, SkLearnEncoder encoder){

				switch(stackMethod){
					case SkLearnMethods.PREDICT:
						break;
					default:
						throw new IllegalArgumentException(stackMethod);
				}

				Feature feature = encoder.exportPrediction(model, FieldNameUtil.create(Estimator.FIELD_PREDICT, index), continuousLabel);

				return Collections.singletonList(feature);
			}
		};

		return StackingUtil.encodeStacking(estimators, stackMethod, predictFunction, finalEstimator, passthrough, schema);
	}

	@Override
	public List<Regressor> getEstimators(){
		return getList("estimators_", Regressor.class);
	}

	public Regressor getFinalEstimator(){
		return getRegressor("final_estimator_");
	}

	public Boolean getPassthrough(){
		return getBoolean("passthrough");
	}

	public List<String> getStackMethod(){
		return getEnumList("stack_method_", this::getStringList, Arrays.asList(SkLearnMethods.PREDICT));
	}
}