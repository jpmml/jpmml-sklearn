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

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.TypeUtil;
import sklearn.HasMultiType;
import sklearn.SkLearnTransformer;

abstract
public class BaseEncoder extends SkLearnTransformer implements HasMultiType {

	public BaseEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public DataType getDataType(int index){
		List<List<Object>> categories = getCategories();

		List<Object> featureCategories = categories.get(index);

		featureCategories = featureCategories.stream()
			.filter(category -> !EncoderUtil.isMissingCategory(category))
			.collect(Collectors.toList());

		return TypeUtil.getDataType(featureCategories, DataType.STRING);
	}

	public List<List<Object>> getCategories(){
		return getArrayList("categories_", Object.class);
	}
}