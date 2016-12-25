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

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.FeatureMapper;

abstract
public class Regressor extends Estimator {

	public Regressor(String module, String name){
		super(module, name);
	}

	@Override
	public boolean isSupervised(){
		return true;
	}

	@Override
	public Schema createSchema(FeatureMapper featureMapper){
		Schema result;

		if(featureMapper.isEmpty()){
			featureMapper.initActiveFields(createActiveFields(getNumberOfFeatures()), getOpType(), getDataType());
			featureMapper.initTargetField(createTargetField(), OpType.CONTINUOUS, DataType.DOUBLE, null);

			result = featureMapper.createSupervisedSchema();
		} else

		{
			featureMapper.updateActiveFields(true, getOpType(), getDataType());
			featureMapper.updateTargetField(OpType.CONTINUOUS, DataType.DOUBLE, null);

			result = featureMapper.createSupervisedSchema();

			if(requiresContinuousInput()){
				result = featureMapper.cast(OpType.CONTINUOUS, getDataType(), result);
			}
		}

		return result;
	}
}