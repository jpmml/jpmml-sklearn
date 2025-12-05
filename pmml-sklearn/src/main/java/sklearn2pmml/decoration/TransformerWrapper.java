/*
 * Copyright (c) 2022 Villu Ruusmann
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
package sklearn2pmml.decoration;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

abstract
public class TransformerWrapper extends Transformer {

	public TransformerWrapper(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		Transformer transformer = getTransformer();

		return transformer.getNumberOfFeatures();
	}

	@Override
	public DataType getDataType(){
		Transformer transformer = getTransformer();

		return transformer.getDataType();
	}

	@Override
	public OpType getOpType(){
		Transformer transformer = getTransformer();

		return transformer.getOpType();
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Transformer transformer = getTransformer();

		return transformer.encode(features, encoder);
	}

	public Transformer getTransformer(){
		return getTransformer("transformer_");
	}
}