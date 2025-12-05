/*
 * Copyright (c) 2023 Villu Ruusmann
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
package pycaret.preprocess;

import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.HasEstimator;
import sklearn.IdentityTransformer;
import sklearn.OutlierDetector;

public class RemoveOutliers extends IdentityTransformer implements HasEstimator<Estimator> {

	public RemoveOutliers(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();

		CastFunction<OutlierDetector> castFunction = new CastFunction<OutlierDetector>(OutlierDetector.class){

			@Override
			public String formatMessage(Object object){
				return "The outlier detector object (" + ClassDictUtil.formatClass(object) + ") is not supported";
			}
		};

		OutlierDetector outlierDetector = castFunction.apply(estimator);

		Schema schema = new Schema(encoder, null, features);

		Model model = estimator.encode(schema);

		encoder.addTransformer(model);

		encoder.export(model, Arrays.asList(outlierDetector.getDecisionFunctionField(), outlierDetector.getOutlierField()));

		Predicate predicate = new SimplePredicate(outlierDetector.getOutlierField(), SimplePredicate.Operator.EQUAL, false);

		encoder.setPredicate(predicate);

		return super.encodeFeatures(features, encoder);
	}

	@Override
	public Estimator getEstimator(){
		return getEstimator("_estimator");
	}
}