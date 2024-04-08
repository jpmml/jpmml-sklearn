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
import java.util.List;

import sklearn.HasPriorProbability;

public class LogitLink extends Link {

	public LogitLink(String module, String name){
		super(module, name);
	}

	@Override
	public List<Double> computeInitialPredictions(int numClasses, HasPriorProbability hasPriorProbability){
		List<Double> result = new AbstractList<Double>(){

			@Override
			public int size(){
				return numClasses;
			}

			@Override
			public Double get(int index){
				Number priorProbability = hasPriorProbability.getPriorProbability(index);

				return logit(priorProbability.doubleValue());
			}
		};

		return result;
	}

	static
	public double logit(double x){
		return Math.log(x / (1d - x));
	}
}