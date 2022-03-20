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

import java.util.ArrayList;
import java.util.List;

import net.razorvine.pickle.objects.ClassDictConstructor;
import numpy.DType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictConstructorUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.CategoricalDtype;

abstract
public class Transformer extends Step {

	public Transformer(String module, String name){
		super(module, name);
	}

	abstract
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder);

	@Override
	public int getNumberOfFeatures(){

		if(containsKey(SkLearnFields.N_FEATURES_IN)){
			return getInteger(SkLearnFields.N_FEATURES_IN);
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public DataType getDataType(){
		return DataType.DOUBLE;
	}

	public List<Feature> encode(List<Feature> features, SkLearnEncoder encoder){
		checkFeatures(features);

		features = updateFeatures(features, encoder);

		return encodeFeatures(features, encoder);
	}

	public void checkFeatures(List<? extends Feature> features){
		StepUtil.checkNumberOfFeatures(this, features);
	}

	public List<Feature> updateFeatures(List<Feature> features, SkLearnEncoder encoder){
		OpType opType;
		DataType dataType;

		try {
			opType = getOpType();
			dataType = getDataType();
		} catch(UnsupportedOperationException uoe){
			return features;
		}

		List<Feature> result = new ArrayList<>(features.size());

		for(Feature feature : features){

			if(feature instanceof WildcardFeature){
				WildcardFeature wildcardFeature = (WildcardFeature)feature;

				String name = wildcardFeature.getName();

				DataField dataField = encoder.getDataField(name);
				if(dataField == null){
					throw new IllegalArgumentException("Field " + name + " is undefined");
				}

				dataField = updateDataField(dataField, opType, dataType, encoder);

				feature = new WildcardFeature(encoder, dataField);
			}

			result.add(feature);
		}

		return result;
	}

	public DataField updateDataField(DataField dataField, OpType opType, DataType dataType, SkLearnEncoder encoder){
		String name = dataField.requireName();

		if(encoder.isFrozen(name)){
			return dataField;
		}

		switch(dataType){
			case DOUBLE:
				// If the DataField element already specifies a non-default data type, then keep it
				if(dataField.requireDataType() != DataType.DOUBLE){
					dataType = dataField.requireDataType();
				}
				break;
		}

		dataField
			.setOpType(opType)
			.setDataType(dataType);

		return dataField;
	}

	public Object getOptionalDType(String name, boolean extended){
		Object value = get(name);

		if(value == null){
			return null;
		}

		return getDType(name, extended);
	}

	public Object getDType(String name, boolean extended){
		Object dtype = get(name);

		if(dtype instanceof String){
			String string = (String)dtype;

			if(extended){
				return string;
			}
		} else

		if(dtype instanceof CategoricalDtype){
			CategoricalDtype categoricalDtype = (CategoricalDtype)dtype;

			return categoricalDtype.getDType();
		} else

		if(dtype instanceof ClassDictConstructor){
			ClassDictConstructor dictConstructor = (ClassDictConstructor)dtype;

			return ClassDictConstructorUtil.construct(dictConstructor, DType.class);
		} else

		if(dtype instanceof DType){
			DType numpyDType = (DType)dtype;

			return numpyDType;
		}

		return get(name, DType.class);
	}
}