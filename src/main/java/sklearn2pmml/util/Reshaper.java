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
package sklearn2pmml.util;

import java.util.List;

import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.MultiTransformer;

public class Reshaper extends MultiTransformer {

	public Reshaper(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		Object[] newshape = getNewshape();

		if(newshape.length >= 2){
			return ValueUtil.asInteger((Number)newshape[1]);
		}

		return super.getNumberOfFeatures();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		return features;
	}

	public Object[] getNewshape(){
		return getTuple("newshape");
	}
}