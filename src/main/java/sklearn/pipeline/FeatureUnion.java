/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.pipeline;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TupleUtil;
import sklearn.Transformer;
import sklearn.TransformerUtil;

public class FeatureUnion extends Transformer {

	public FeatureUnion(String module, String name){
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
		List<Transformer> transformers = getTransformers();

		List<Feature> result = new ArrayList<>();

		for(Transformer transformer : transformers){
			encoder.updateFeatures(features, transformer);

			List<Feature> transformerFeatures = new ArrayList<>(features);

			transformerFeatures = transformer.encodeFeatures(transformerFeatures, encoder);

			result.addAll(transformerFeatures);
		}

		return result;
	}

	public List<Transformer> getTransformers(){
		List<Object[]> transformerList = getTransformerList();

		return TransformerUtil.asTransformerList(TupleUtil.extractElementList(transformerList, 1));
	}

	public List<Object[]> getTransformerList(){
		return (List)get("transformer_list");
	}
}