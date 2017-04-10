/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn.ensemble;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import sklearn.Classifier;
import sklearn.EstimatorUtil;

abstract
public class EnsembleClassifier extends Classifier {

	public EnsembleClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		Classifier baseEstimator = getBaseEstimator();

		return baseEstimator.getOpType();
	}

	@Override
	public DataType getDataType(){
		Classifier baseEstimator = getBaseEstimator();

		return baseEstimator.getDataType();
	}

	public Classifier getBaseEstimator(){
		Object baseEstimator = get("base_estimator_");

		return EstimatorUtil.asClassifier(baseEstimator);
	}

	public List<? extends Classifier> getEstimators(){
		List<?> estimators = (List)get("estimators_");

		return EstimatorUtil.asClassifierList(estimators);
	}
}