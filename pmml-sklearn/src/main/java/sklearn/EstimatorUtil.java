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
package sklearn;

import java.util.List;

import org.jpmml.python.ClassDictUtil;

public class EstimatorUtil {

	private EstimatorUtil(){
	}

	static
	public List<?> getClasses(Estimator estimator){

		if(estimator instanceof HasClasses){
			HasClasses hasClasses = (HasClasses)estimator;

			return hasClasses.getClasses();
		}

		throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(estimator) + ") is not a classifier");
	}

	static
	public int getNumberOfOutputs(int[] shape){

		// (n_samples)
		if(shape.length == 1){
			return 1;
		} else

		// (n_samples, n_outputs)
		if(shape.length == 2){
			return shape[1];
		} else

		{
			throw new IllegalArgumentException();
		}
	}
}
