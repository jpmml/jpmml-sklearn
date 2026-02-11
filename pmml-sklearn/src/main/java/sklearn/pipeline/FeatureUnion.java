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
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import numpy.core.ScalarUtil;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ProductFeature;
import org.jpmml.python.CastFunction;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasHead;
import sklearn.SkLearnTransformer;
import sklearn.SkLearnTransformerCastFunction;
import sklearn.Step;
import sklearn.StepUtil;
import sklearn.Transformer;

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
		Map<String, ?> transformerWeights = getTransformerWeights();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < transformers.size(); i++){
			String name = getName(transformers.get(i));
			Transformer transformer = getTransformer(transformers.get(i));

			List<Feature> transformerFeatures = new ArrayList<>(features);

			transformerFeatures = transformer.encode(transformerFeatures, encoder);

			if(transformerWeights != null && !transformerWeights.isEmpty()){
				Number weight = (Number)ScalarUtil.decode(transformerWeights.get(name));

				if(weight == null){
					throw new IllegalArgumentException(name);
				}

				transformerFeatures = transformerFeatures.stream()
					.map(transformerFeature -> {
						return new ProductFeature(encoder, transformerFeature, weight){

							@Override
							public ContinuousFeature toContinuousFeature(){
								Feature feature = getFeature();
								Number factor = getFactor();

								ContinuousFeature continuousFeature = feature.toContinuousFeature();

								if(factor.doubleValue() == 1d){
									return continuousFeature;
								} else

								{
									Supplier<Apply> applySupplier = () -> {
										return ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, continuousFeature.ref(), ExpressionUtil.createConstant(factor));
									};

									return toContinuousFeature(FieldNameUtil.create("weighted", transformerFeature), DataType.DOUBLE, applySupplier);
								}
							}
						};
					})
					.collect(Collectors.toList());
			}

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

	public Map<String, ?> getTransformerWeights(){
		return getOptionalDict("transformer_weights");
	}

	static
	protected String getName(Object[] transformer){
		return TupleUtil.extractStringElement(transformer, 0);
	}

	static
	protected Transformer getTransformer(Object[] transformer){
		CastFunction<Transformer> castFunction = new SkLearnTransformerCastFunction();

		return TupleUtil.extractElement(transformer, 1, castFunction);
	}
}