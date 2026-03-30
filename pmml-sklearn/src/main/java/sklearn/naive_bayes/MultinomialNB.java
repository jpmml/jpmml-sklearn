/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn.naive_bayes;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.jpmml.converter.CMatrixUtil;

public class MultinomialNB extends ContinuousNB {

	public MultinomialNB(String module, String name){
		super(module, name);
	}

	@Override
	public List<Number> getCoefficients(List<Number> featureLogProb, int index, int numberOfClasses, int numberOfFeatures){
		List<Number> coefficients = CMatrixUtil.getRow(featureLogProb, numberOfClasses, numberOfFeatures, index);

		Function<Number, Number> function = new Function<Number, Number>(){

			@Override
			public Number apply(Number value){

				if(value.doubleValue() == Double.NEGATIVE_INFINITY){
					return null;
				}

				return value;
			}
		};

		return Lists.transform(coefficients, function);
	}
}