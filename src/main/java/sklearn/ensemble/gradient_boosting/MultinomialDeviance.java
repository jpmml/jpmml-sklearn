/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.ensemble.gradient_boosting;

import java.util.ArrayList;
import java.util.List;

import org.jpmml.converter.ExpTransformation;
import sklearn.HasPriorProbability;

public class MultinomialDeviance extends LossFunction {

	public MultinomialDeviance(String module, String name){
		super(module, name);
	}

	@Override
	public List<? extends Number> computeInitialPredictions(HasPriorProbability init){
		Integer k = getK();

		List<Double> result = new ArrayList<>();

		for(int i = 0; i < k; i++){
			Number proba = init.getPriorProbability(i);

			result.add(Math.log(proba.doubleValue()));
		}

		return result;
	}

	@Override
	public ExpTransformation createTransformation(){
		return new ExpTransformation();
	}
}