/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn2pmml;

import java.util.List;

import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasClasses;

public class EstimatorProxy extends Estimator implements HasClasses {

	public EstimatorProxy(String module, String name){
		super(module, name);
	}

	@Override
	public Object get(Object key){

		if(super.containsKey(key)){
			return super.get(key);
		}

		Estimator estimator = getEstimator();

		return estimator.get(key);
	}

	@Override
	public MiningFunction getMiningFunction(){
		Estimator estimator = getEstimator();

		return estimator.getMiningFunction();
	}

	@Override
	public boolean isSupervised(){
		Estimator estimator = getEstimator();

		return estimator.isSupervised();
	}

	@Override
	public int getNumberOfFeatures(){
		Estimator estimator = getEstimator();

		return estimator.getNumberOfFeatures();
	}

	@Override
	public OpType getOpType(){
		Estimator estimator = getEstimator();

		return estimator.getOpType();
	}

	@Override
	public DataType getDataType(){
		Estimator estimator = getEstimator();

		return estimator.getDataType();
	}

	@Override
	public List<?> getClasses(){
		Estimator estimator = getEstimator();

		return EstimatorUtil.getClasses(estimator);
	}

	@Override
	public Model encodeModel(Schema schema){
		throw new UnsupportedOperationException();
	}

	@Override
	public Model encodeModel(Schema schema, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();

		return estimator.encodeModel(schema, encoder);
	}

	public Estimator getEstimator(){
		ClassDict estimator = (ClassDict)super.get("estimator_");

		return EstimatorUtil.asEstimator(estimator);
	}
}