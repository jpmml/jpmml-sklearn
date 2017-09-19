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

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.TreeModelProducer;
import sklearn.Regressor;
import sklearn.tree.DecisionTreeRegressor;

public class GradientBoostingRegressor extends Regressor implements TreeModelProducer {

	public GradientBoostingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){

		// SkLearn 0.18
		if(containsKey("n_features")){
			return ValueUtil.asInt((Number)get("n_features"));
		}

		// SkLearn 0.19+
		return super.getNumberOfFeatures();
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		HasDefaultValue init = getInit();

		List<DecisionTreeRegressor> estimators = getEstimators();

		return GradientBoostingUtil.encodeGradientBoosting(estimators, init.getDefaultValue(), getLearningRate(), schema);
	}

	public HasDefaultValue getInit(){
		Object init = get("init_");

		try {
			if(init == null){
				throw new NullPointerException();
			}

			return (HasDefaultValue)init;
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(init) + ") is not a BaseEstimator or is not a supported BaseEstimator subclass", re);
		}
	}

	public Number getLearningRate(){
		return (Number)get("learning_rate");
	}

	public List<DecisionTreeRegressor> getEstimators(){
		return (List)ClassDictUtil.getArray(this, "estimators_");
	}
}