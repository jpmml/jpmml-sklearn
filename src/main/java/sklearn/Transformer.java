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
import org.jpmml.sklearn.ClassDictConstructorUtil;
import org.jpmml.sklearn.PyClassDict;
import org.jpmml.sklearn.SkLearnEncoder;

abstract
public class Transformer extends PyClassDict {

	public Transformer(String module, String name){
		super(module, name);
	}

	abstract
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder);

	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	public DataType getDataType(){
		return DataType.DOUBLE;
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

				DataField dataField = encoder.updateDataField(wildcardFeature.getName(), opType, dataType);

				feature = new WildcardFeature(encoder, dataField);
			}

			result.add(feature);
		}

		return result;
	}

	public List<Feature> updateAndEncodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		features = updateFeatures(features, encoder);

		return encodeFeatures(features, encoder);
	}

	public DType getDType(){
		Object dtype = getObject("dtype");

		if(dtype instanceof ClassDictConstructor){
			ClassDictConstructor classDictConstructor = (ClassDictConstructor)dtype;

			dtype = ClassDictConstructorUtil.construct(classDictConstructor, DType.class);
		}

		return (DType)dtype;
	}
}