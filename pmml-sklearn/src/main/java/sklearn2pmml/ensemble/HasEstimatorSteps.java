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
package sklearn2pmml.ensemble;

import java.util.List;

import com.google.common.collect.Lists;
import org.jpmml.python.CastFunction;
import org.jpmml.python.TupleUtil;
import sklearn.Estimator;
import sklearn.EstimatorCastFunction;
import sklearn.HasSteps;

public interface HasEstimatorSteps extends HasSteps {

	default
	List<? extends Estimator> getEstimators(){
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException();
		}

		List<?> estimators = TupleUtil.extractElementList(steps, 1);

		CastFunction<Estimator> castFunction = new EstimatorCastFunction<Estimator>(Estimator.class);

		return Lists.transform(estimators, castFunction);
	}
}