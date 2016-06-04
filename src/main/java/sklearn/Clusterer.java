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

import org.dmg.pmml.Model;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sklearn.FeatureMapper;

abstract
public class Clusterer extends Estimator {

	public Clusterer(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(FeatureMapper featureMapper){

		if(featureMapper.isEmpty()){
			featureMapper.initActiveFields(SchemaUtil.createActiveFields(getNumberOfFeatures()), getOpType(), getDataType());
		} else

		{
			if(requiresContinuousInput()){
				featureMapper.simplifyActiveFields(false, getOpType(), getDataType());
			}

			featureMapper.updateActiveFields(getNumberOfFeatures(), false, getOpType(), getDataType());
		}

		FeatureSchema schema = featureMapper.createUnsupervisedSchema();

		return encodeModel(schema);
	}
}