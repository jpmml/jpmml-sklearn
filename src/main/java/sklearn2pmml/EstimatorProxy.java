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

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.CastFunction;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.PyClassDict;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasClasses;
import sklearn.HasEstimator;

public class EstimatorProxy extends Estimator implements HasClasses, HasEstimator<Estimator> {

	public EstimatorProxy(){
		super("sklearn2pmml", "EstimatorProxy");
	}

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

		return ClassifierUtil.getClasses(estimator);
	}

	@Override
	public Model encodeModel(Schema schema){
		Estimator estimator = getEstimator();

		return estimator.encodeModel(schema);
	}

	/**
	 * @see PyClassDict#get(String, Class)
	 */
	@Override
	public Estimator getEstimator(){
		Object estimator = super.get("estimator_");

		if(estimator == null){
			throw new IllegalArgumentException("Attribute \'" + ClassDictUtil.formatMember(this, "estimator_") + "\' has a missing (None/null) value");
		}

		CastFunction<Estimator> castFunction = new CastFunction<Estimator>(Estimator.class){

			@Override
			protected String formatMessage(Object object){
				return "Attribute \'" + ClassDictUtil.formatMember(EstimatorProxy.this, "estimator_") + "\' has an unsupported value (" + ClassDictUtil.formatClass(object) + ")";
			}
		};

		return castFunction.apply(estimator);
	}
}