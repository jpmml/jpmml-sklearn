/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sklearn.loss;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;

import sklearn.HasPriorProbability;

public class MultinomialLogit extends Link {

	public MultinomialLogit(String module, String name){
		super(module, name);
	}

	@Override
	public List<? extends Number> computeInitialPredictions(int numClasses, HasPriorProbability hasPriorProbability){
		List<Number> priorProbabilities = new ArrayList<>();

		for(int i = 0; i < numClasses; i++){
			priorProbabilities.add(hasPriorProbability.getPriorProbability(i));
		}

		double gmean = gmean(priorProbabilities);

		List<Double> result = new AbstractList<Double>(){

			@Override
			public int size(){
				return numClasses;
			}

			@Override
			public Double get(int index){
				Number priorProbability = priorProbabilities.get(index);

				return Math.log(priorProbability.doubleValue() / gmean);
			}
		};

		return result;
	}

	static
	public double gmean(List<? extends Number> values){
		double logSum = 0d;

		for(Number value : values){
			double doubleValue = value.doubleValue();

			if(doubleValue == 0d){
				return Double.NaN;
			}

			logSum += Math.log(doubleValue);
		}

		return Math.exp(logSum / values.size());
	}
}