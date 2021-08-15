/*
 * Copyright (c) 2021 Villu Ruusmann
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

import java.util.List;
import java.util.stream.Collectors;

import numpy.DType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.HasArray;
import sklearn.MultiTransformer;

abstract
public class BaseEncoder extends MultiTransformer {

	public BaseEncoder(String module, String name){
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

			featureCategories = featureCategories.stream()
				.filter(value -> (value != null) && !ValueUtil.isNaN(value))
				.collect(Collectors.toList());

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

	public List<List<?>> getCategories(){
		return EncoderUtil.transformCategories(getList("categories_", HasArray.class));
	}

	public DType getDType(){
		return (DType)getDType(false);
	}

	public String getHandleUnknown(){
		return getOptionalString("handle_unknown");
	}
}