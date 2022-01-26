/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.model_selection;

import org.jpmml.python.Castable;
import org.jpmml.python.PythonObject;
import sklearn.Estimator;

public class EstimatorSearcher extends PythonObject implements Castable {

	public EstimatorSearcher(String module, String name){
		super(module, name);
	}

	@Override
	public Object castTo(Class<?> clazz){

		if((Estimator.class).isAssignableFrom(clazz)){
			Class<? extends Estimator> estimatorClazz = clazz.asSubclass(Estimator.class);

			return getBestEstimator(estimatorClazz);
		}

		return this;
	}

	public Estimator getBestEstimator(){
		return getBestEstimator(Estimator.class);
	}

	public Estimator getBestEstimator(Class<? extends Estimator> clazz){
		return get("best_estimator_", clazz);
	}
}