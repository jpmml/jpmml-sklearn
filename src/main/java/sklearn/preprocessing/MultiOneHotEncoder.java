/*
 * Copyright (c) 2018 Villu Ruusmann
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

import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.HasArray;

public class MultiOneHotEncoder extends OneHotEncoder {

	public MultiOneHotEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public List<? extends Number> getValues(){
		List<? extends HasArray> categories = getCategories();

		ClassDictUtil.checkSize(1, categories);

		HasArray hasArray = categories.get(0);

		// XXX
		return (List)hasArray.getArrayContent();
	}

	@Override
	public List<? extends Number> getActiveFeatures(){
		throw new UnsupportedOperationException();
	}

	public List<? extends HasArray> getCategories(){
		return get("categories_", List.class);
	}

	@Override
	public List<Integer> getFeatureSizes(){
		throw new UnsupportedOperationException();
	}
}