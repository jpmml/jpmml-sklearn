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
package sklearn.feature_selection;

import java.util.ArrayList;
import java.util.List;

import net.razorvine.pickle.objects.ClassDict;
import numpy.core.Scalar;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.Selector;

public class SelectFromModel extends Selector {

	public SelectFromModel(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		Estimator estimator = getEstimator();

		return estimator.getNumberOfFeatures();
	}

	@Override
	public List<Boolean> getSupportMask(){
		Estimator estimator = getEstimator();
		Number threshold = getThreshold();

		List<? extends Number> featureImportances = (List)ClassDictUtil.getArray(estimator, "feature_importances_");
		if(featureImportances == null){
			throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(estimator) + ") does not have a persistent \'feature_importances_\' attribute");
		}

		List<Boolean> result = new ArrayList<>();

		for(int i = 0; i < featureImportances.size(); i++){
			Number featureImportance = featureImportances.get(i);

			result.add(featureImportance.doubleValue() >= threshold.doubleValue());
		}

		return result;
	}

	public Estimator getEstimator(){
		ClassDict estimator = (ClassDict)get("estimator_");

		return EstimatorUtil.asEstimator(estimator);
	}

	public Number getThreshold(){
		Scalar threshold = (Scalar)get("threshold_");

		// SkLearn 0.19+
		if(threshold == null){
			throw new IllegalArgumentException("The selector object does not have a persistent \'threshold_\' attribute");
		}

		return (Number)threshold.getOnlyElement();
	}
}