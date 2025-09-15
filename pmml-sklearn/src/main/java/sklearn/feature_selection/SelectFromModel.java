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

import org.jpmml.python.AttributeException;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnException;
import sklearn.Estimator;
import sklearn.HasEstimator;
import sklearn2pmml.EstimatorProxy;
import sklearn2pmml.SelectorProxy;

public class SelectFromModel extends SkLearnSelector implements HasEstimator<Estimator> {

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

		List<Number> featureImportances;

		try {
			featureImportances = estimator.getNumberArray("feature_importances_");
		} catch(AttributeException ae){
			String message = "The estimator object (" + ClassDictUtil.formatClass(estimator) + ") does not have a persistent \'feature_importances_\' attribute. " +
				"Please use the " + (EstimatorProxy.class).getName() + " wrapper class to give this estimator object a persistent state (eg. " + EstimatorProxy.formatProxyExample(estimator) +")";

			throw new SkLearnException(message, ae);
		}

		List<Boolean> result = new ArrayList<>();

		for(int i = 0; i < featureImportances.size(); i++){
			Number featureImportance = featureImportances.get(i);

			result.add(featureImportance.doubleValue() >= threshold.doubleValue());
		}

		return result;
	}

	@Override
	public Estimator getEstimator(){
		return get("estimator_", Estimator.class);
	}

	public Number getThreshold(){
		Number threshold;

		// SkLearn 0.19+
		try {
			threshold = getNumber("threshold_");
		} catch(AttributeException ae){
			String message = "The selector object (" + ClassDictUtil.formatClass(this) + ") does not have a persistent \'threshold_\' attribute. " +
				"Please use the " + (SelectorProxy.class).getName() + " wrapper class to give this selector object a persistent state (eg. " + SelectorProxy.formatProxyExample(this) + ")";

			throw new SkLearnException(message, ae);
		}

		return threshold;
	}
}