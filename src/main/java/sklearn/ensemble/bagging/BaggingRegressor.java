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
package sklearn.ensemble.bagging;

import java.util.List;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.Regressor;

public class BaggingRegressor extends Regressor {

	public BaggingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public boolean requiresContinuousInput(){
		Regressor baseEstimator = getBaseEstimator();

		return baseEstimator.requiresContinuousInput();
	}

	@Override
	public DataType getDataType(){
		Regressor baseEstimator = getBaseEstimator();

		return baseEstimator.getDataType();
	}

	@Override
	public OpType getOpType(){
		Regressor baseEstimator = getBaseEstimator();

		return baseEstimator.getOpType();
	}

	@Override
	public MiningModel encodeModel(FeatureSchema schema){
		List<Regressor> estimators = getEstimators();
		List<List<Integer>> estimatorsFeatures = getEstimatorsFeatures();

		MiningModel miningModel = BaggingUtil.encodeBagging(estimators, estimatorsFeatures, MultipleModelMethodType.AVERAGE, MiningFunctionType.REGRESSION, schema);

		return miningModel;
	}

	@Override
	public Set<DefineFunction> encodeDefineFunctions(){
		Regressor baseEstimator = getBaseEstimator();

		return baseEstimator.encodeDefineFunctions();
	}

	public Regressor getBaseEstimator(){
		Object baseEstimator = get("base_estimator_");

		return BaggingRegressor.transformer.apply(baseEstimator);
	}

	public List<Regressor> getEstimators(){
		List<?> estimators = (List)get("estimators_");

		return Lists.transform(estimators, BaggingRegressor.transformer);
	}

	public List<List<Integer>> getEstimatorsFeatures(){
		List<?> estimatorsFeatures = (List)get("estimators_features_");

		return BaggingUtil.transformEstimatorsFeatures(estimatorsFeatures);
	}

	private static final Function<Object, Regressor> transformer = new Function<Object, Regressor>(){

		@Override
		public Regressor apply(Object object){

			try {
				if(object == null){
					throw new NullPointerException();
				}

				return (Regressor)object;
			} catch(RuntimeException re){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Regressor or is not a supported Regressor subclass");
			}
		}
	};
}