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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import numpy.DType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class OrdinalEncoder extends Transformer {

	public OrdinalEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		List<List<?>> categories = getCategories();

		DataType result = null;

		for(int i = 0; i < categories.size(); i++){
			List<?> featureCategories = categories.get(i);

			DataType dataType = TypeUtil.getDataType(featureCategories, null);

			if(result == null){
				result = dataType;
			} else

			{
				if(!(result).equals(dataType)){
					throw new UnsupportedOperationException();
				}
			}
		}

		if(result == null){
			result = DataType.STRING;
		}

		return result;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<List<?>> categories = getCategories();
		DType dtype = getDType();

		ClassDictUtil.checkSize(categories, features);

		List<Feature> result = new ArrayList<>();

		DataType dataType = dtype.getDataType();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			List<?> featureCategories = categories.get(i);

			result.add(EncoderUtil.encodeIndexFeature(this, feature, featureCategories, dataType, encoder));
		}

		return result;
	}

	public List<List<?>> getCategories(){
		return EncoderUtil.transformCategories(getList("categories_", HasArray.class));
	}

	public DType getDType(){
		return (DType)getDType(false);
	}
}