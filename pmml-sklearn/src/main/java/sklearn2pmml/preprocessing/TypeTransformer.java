/*
 * Copyright (c) 2025 Villu Ruusmann
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

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.CategoricalDtypeUtil;
import pandas.core.CategoricalDtype;
import sklearn.Transformer;

abstract
public class TypeTransformer extends Transformer {

	public TypeTransformer(String module, String name){
		super(module, name);
	}

	public Feature refineFeature(Feature feature, TypeInfo dtype, SkLearnEncoder encoder){
		DataType dataType = dtype.getDataType();
		OpType opType = TypeUtil.getOpType(dataType);

		feature = refineFeature(feature, opType, dataType, encoder);

		if(dtype instanceof CategoricalDtype){
			CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

			feature = CategoricalDtypeUtil.refineFeature(feature, categoricalDtype, encoder);
		}

		return feature;
	}

	public Feature refineFeature(Feature feature, OpType opType, DataType dataType, SkLearnEncoder encoder){

		if(feature instanceof WildcardFeature){
			WildcardFeature wildcardFeature = (WildcardFeature)feature;

			wildcardFeature = refineWildcardFeature(wildcardFeature, opType, dataType, encoder);

			DataField dataField = wildcardFeature.getField();

			switch(opType){
				case CONTINUOUS:
					return new ContinuousFeature(encoder, dataField);
				case CATEGORICAL:
				case ORDINAL:
					return new ObjectFeature(encoder, dataField);
				default:
					throw new IllegalArgumentException();
			}
		} else

		{
			if(feature.getDataType() != dataType){
				FieldRef fieldRef = feature.ref();

				DerivedField derivedField = encoder.ensureDerivedField(createFieldName((dataType.name()).toLowerCase(), feature), opType, dataType, () -> fieldRef);

				return FeatureUtil.createFeature(derivedField, encoder);
			}
		}

		return feature;
	}
}