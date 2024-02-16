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
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.CategoricalDtypeUtil;
import pandas.core.CategoricalDtype;
import sklearn.Transformer;

public class CastTransformer extends Transformer {

	public CastTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		TypeInfo dtype = getDType();

		DataType dataType = dtype.getDataType();
		OpType opType = TypeUtil.getOpType(dataType);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if(feature instanceof WildcardFeature){
				WildcardFeature wildcardFeature = (WildcardFeature)feature;

				feature = refineWildcardFeature(wildcardFeature, opType, dataType, encoder);
			} else

			{
				if(feature.getDataType() != dataType){
					FieldRef fieldRef = feature.ref();

					DerivedField derivedField = encoder.ensureDerivedField(createFieldName((dataType.name()).toLowerCase(), feature), opType, dataType, () -> fieldRef);

					feature = FeatureUtil.createFeature(derivedField, encoder);
				}
			} // End if

			if(dtype instanceof CategoricalDtype){
				CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

				feature = CategoricalDtypeUtil.refineFeature(feature, categoricalDtype, encoder);
			}

			result.add(feature);
		}

		return result;
	}

	public TypeInfo getDType(){

		// SkLearn2PMML 0.101.0+
		if(containsKey("dtype_")){
			return getDType("dtype_", true);
		} else

		// SkLearn2PMML 0.100.2
		{
			return getDType("dtype", true);
		}
	}
}