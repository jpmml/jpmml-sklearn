/*
 * Copyright (c) 2020 Villu Ruusmann
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
package sklearn2pmml.ensemble;

import java.util.List;
import java.util.Objects;

import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.InvalidLabelException;
import org.jpmml.converter.Schema;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.Transformer;
import sklearn2pmml.HasController;

public class SelectFirstClassifier extends Classifier implements HasController, HasEstimatorSteps {

	public SelectFirstClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		List<? extends Estimator> estimators = getEstimators();

		List<?> result = null;

		for(Estimator estimator : estimators){

			if(result == null){
				result = EstimatorUtil.getClasses(estimator);
			} else

			{
				if(!Objects.equals(result, EstimatorUtil.getClasses(estimator))){
					throw new InvalidLabelException("Expected the same target categories, got " + result + " and " + EstimatorUtil.getClasses(estimator));
				}
			}
		}

		return result;
	}

	@Override
	public boolean hasProbabilityDistribution(){
		List<? extends Estimator> estimators = getEstimators();

		boolean result = true;

		for(Estimator estimator : estimators){
			result &= EstimatorUtil.hasProbabilityDistribution(estimator);
		}

		return result;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		return SelectFirstUtil.encodeSelectFirstEstimator(this, schema);
	}

	@Override
	public Transformer getController(){
		return getOptionalTransformer("controller");
	}

	@Override
	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}
}