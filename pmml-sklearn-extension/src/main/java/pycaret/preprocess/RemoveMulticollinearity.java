/*
 * Copyright (c) 2023 Villu Ruusmann
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
package pycaret.preprocess;

import java.util.List;
import java.util.stream.Collectors;

import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.MultiTransformer;

public class RemoveMulticollinearity extends MultiTransformer {

	public RemoveMulticollinearity(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<String> drop = getDrop();

		List<Feature> result = features.stream()
			.filter(feature -> !drop.contains(feature.getName()))
			.collect(Collectors.toList());

		return result;
	}

	public List<String> getDrop(){
		return getList("_drop", String.class);
	}
}