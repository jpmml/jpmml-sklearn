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
import org.jpmml.converter.Feature;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn.TransformerUtil;

public class CastTransformer extends Transformer {

	public CastTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		TypeInfo dtype = getDType();

		DataType dataType = dtype.getDataType();
		OpType opType = TransformerUtil.getOpType(dataType);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if(feature.getDataType() != dataType){
				FieldRef fieldRef = feature.ref();

				DerivedField derivedField = encoder.ensureDerivedField(createFieldName(dataType, feature), opType, dataType, () -> fieldRef);

				feature = TransformerUtil.createFeature(derivedField, encoder);
			}

			result.add(feature);
		}

		return result;
	}

	public TypeInfo getDType(){
		return getDType("dtype", true);
	}
}