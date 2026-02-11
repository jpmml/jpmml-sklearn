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
import org.jpmml.python.CastFunction;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasHead;
import sklearn.SkLearnTransformer;
import sklearn.Step;
import sklearn.StepUtil;
import sklearn.Transformer;
import sklearn.TransformerCastFunction;

public class FeatureUnion extends SkLearnTransformer implements HasHead {

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
		List<Object[]> transformers = getTransformerList();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < transformers.size(); i++){
			Transformer transformer = getTransformer(transformers.get(i));

			List<Feature> transformerFeatures = new ArrayList<>(features);

			transformerFeatures = transformer.encode(transformerFeatures, encoder);

			result.addAll(transformerFeatures);
		}

		return result;
	}

	@Override
	public Step getHead(){
		List<Object[]> transformers = getTransformerList();

		if(!transformers.isEmpty()){
			Transformer transformer = getTransformer(transformers.get(0));

			return StepUtil.getHead(transformer);
		}

		throw new UnsupportedOperationException();
	}

	public List<Object[]> getTransformerList(){
		return getTupleList("transformer_list");
	}

	static
	protected Transformer getTransformer(Object[] transformer){
		CastFunction<Transformer> castFunction = new TransformerCastFunction<Transformer>(Transformer.class);

		return TupleUtil.extractElement(transformer, 1, castFunction);
	}
}