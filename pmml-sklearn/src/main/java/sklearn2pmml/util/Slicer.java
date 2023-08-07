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
package sklearn2pmml.util;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.python.SliceUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class Slicer extends Transformer {

	public Slicer(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public DataType getDataType(){
		throw new UnsupportedOperationException();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Integer start = getStart();
		Integer stop = getStep();
		Integer step = getStep();

		return SliceUtil.slice(features, start, stop, step);
	}

	public Integer getStart(){
		return getOptionalInteger("start");
	}

	public Integer getStop(){
		return getOptionalInteger("stop");
	}

	public Integer getStep(){
		return getOptionalInteger("step");
	}

	private Integer getOptionalInteger(String name){
		return getOptional(name, Integer.class);
	}
}