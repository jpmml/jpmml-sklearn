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
package sklearn2pmml.preprocessing;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.StringFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.StepUtil;
import sklearn.Transformer;

public class CastTransformer extends Transformer {

	public CastTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Object dtype = getDType();

		OpType opType = StepUtil.getOpType(dtype);
		DataType dataType = StepUtil.getDataType(dtype);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if(!(dataType).equals(feature.getDataType())){
				FieldRef fieldRef = feature.ref();

				DerivedField derivedField = encoder.ensureDerivedField(createFieldName((dataType.name()).toLowerCase(), feature), opType, dataType, () -> fieldRef);

				switch(dataType){
					case STRING:
						feature = new StringFeature(encoder, derivedField);
						break;
					case INTEGER:
					case FLOAT:
					case DOUBLE:
						feature = new ContinuousFeature(encoder, derivedField);
						break;
					case BOOLEAN:
					case DATE:
					case DATE_TIME:
						// Falls through
					default:
						feature = new ObjectFeature(encoder, derivedField);
						break;
				}
			}

			result.add(feature);
		}

		return result;
	}

	public Object getDType(){
		return getDType(true);
	}
}