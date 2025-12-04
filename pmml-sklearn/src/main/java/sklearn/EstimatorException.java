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
package sklearn;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnException;

public class EstimatorException extends SkLearnException {

	public EstimatorException(Estimator estimator, Class<?>... clazzes){
		super("The estimator object (" + ClassDictUtil.formatClass(estimator) + ") is not an instance of " + formatClasses(clazzes));
	}

	static
	private Object formatClasses(Class<?>... clazzes){

		if(clazzes.length == 1){
			Class<?> clazz = clazzes[0];

			return clazz.getName();
		} else

		{
			return Arrays.stream(clazzes)
				.map(clazz -> clazz.getName())
				.collect(Collectors.toList());
		}
	}
}