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

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.Schema;
import sklearn.EstimatorUtil;
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
	public MiningModel encodeModel(Schema schema){
		List<? extends Regressor> estimators = getEstimators();
		List<List<Integer>> estimatorsFeatures = getEstimatorsFeatures();

		MiningModel miningModel = BaggingUtil.encodeBagging(estimators, estimatorsFeatures, Segmentation.MultipleModelMethod.AVERAGE, MiningFunction.REGRESSION, schema);

		return miningModel;
	}

	@Override
	public Set<DefineFunction> encodeDefineFunctions(){
		Regressor baseEstimator = getBaseEstimator();

		return baseEstimator.encodeDefineFunctions();
	}

	public Regressor getBaseEstimator(){
		Object baseEstimator = get("base_estimator_");

		return EstimatorUtil.asRegressor(baseEstimator);
	}

	public List<? extends Regressor> getEstimators(){
		List<?> estimators = (List)get("estimators_");

		return EstimatorUtil.asRegressorList(estimators);
	}

	public List<List<Integer>> getEstimatorsFeatures(){
		List<?> estimatorsFeatures = (List)get("estimators_features_");

		return BaggingUtil.transformEstimatorsFeatures(estimatorsFeatures);
	}
}