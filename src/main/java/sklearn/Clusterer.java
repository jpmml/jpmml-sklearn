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

import org.dmg.pmml.OpType;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.FeatureMapper;

abstract
public class Clusterer extends Estimator {

	public Clusterer(String module, String name){
		super(module, name);
	}

	@Override
	public boolean isSupervised(){
		return false;
	}

	@Override
	public Schema createSchema(FeatureMapper featureMapper){

		if(featureMapper.isEmpty()){
			featureMapper.initActiveFields(createActiveFields(getNumberOfFeatures()), getOpType(), getDataType());
		} else

		{
			featureMapper.updateActiveFields(false, getOpType(), getDataType());
		}

		Schema schema = featureMapper.createUnsupervisedSchema();

		if(requiresContinuousInput()){
			schema = featureMapper.cast(OpType.CONTINUOUS, getDataType(), schema);
		}

		return schema;
	}
}