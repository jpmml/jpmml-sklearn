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
package sklearn.linear_model;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BaseLinearUtil {

	private BaseLinearUtil(){
	}

	static
	public RegressionTable encodeRegressionTable(List<Feature> features, Number intercept, List<? extends Number> coefficients){
		ClassDictUtil.checkSize(features, coefficients);

		List<Feature> unusedFeatures = new ArrayList<>();

		List<Double> featureCoefficients = new ArrayList<>(ValueUtil.asDoubles(coefficients));

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Double featureCoefficient = featureCoefficients.get(i);

			if(ValueUtil.isZero(featureCoefficient)){
				featureCoefficients.set(i, Double.NaN);

				unusedFeatures.add(feature);
			}
		}

		if(!unusedFeatures.isEmpty()){
			logger.info("Skipped {} feature(s): {}", unusedFeatures.size(), ClassDictUtil.formatFeatureList(unusedFeatures));
		}

		return RegressionModelUtil.createRegressionTable(features, ValueUtil.asDouble(intercept), featureCoefficients);
	}

	private static final Logger logger = LoggerFactory.getLogger(BaseLinearUtil.class);
}