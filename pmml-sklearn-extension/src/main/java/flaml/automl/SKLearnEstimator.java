/*
 * Copyright (c) 2026 Villu Ruusmann
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
package flaml.automl;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.python.Castable;
import sklearn.Estimator;
import sklearn.HasNumberOfFeatures;
import sklearn.Step;

public class SKLearnEstimator extends Step implements Castable {

	public SKLearnEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public OpType getOpType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public DataType getDataType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public Object castTo(Class<?> clazz){

		if((Step.class).isAssignableFrom(clazz)){
			Class<? extends Step> stepClazz = clazz.asSubclass(Step.class);

			return getModel(stepClazz);
		}

		return this;
	}

	public Estimator getModel(){
		return getModel(Estimator.class);
	}

	public <E extends Step> E getModel(Class<? extends E> clazz){
		return getStep("_model", clazz);
	}
}