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

import java.util.List;

import com.google.common.base.CaseFormat;
import net.razorvine.pickle.objects.ClassDict;
import net.razorvine.pickle.objects.ClassDictConstructor;
import numpy.DType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.sklearn.ClassDictConstructorUtil;
import org.jpmml.sklearn.SkLearnEncoder;

abstract
public class Transformer extends ClassDict {

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

	public DType getDType(){
		Object dtype = get("dtype");

		if(dtype instanceof ClassDictConstructor){
			ClassDictConstructor classDictConstructor = (ClassDictConstructor)dtype;

			dtype = ClassDictConstructorUtil.construct(classDictConstructor, DType.class);
		}

		return (DType)dtype;
	}

	protected String name(){
		Class<? extends Transformer> clazz = getClass();

		String name = clazz.getSimpleName();
		if(name.startsWith("PMML")){
			name = name.substring("PMML".length());
		}

		return CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, name);
	}

	protected FieldName createName(Feature feature){
		return FeatureUtil.createName(name(), feature);
	}

	protected FieldName createName(Feature feature, int index){
		return FeatureUtil.createName(name(), feature, index);
	}
}