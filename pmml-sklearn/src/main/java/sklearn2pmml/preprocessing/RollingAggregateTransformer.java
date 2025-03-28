/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn2pmml.preprocessing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.BlockIndicator;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.Lag;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class RollingAggregateTransformer extends Transformer {

	public RollingAggregateTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String function = getFunction();
		Integer n = getN();
		List<Object> blockIndicators = getBlockIndicators();

		Lag.Aggregate aggregate = parseFunction(function);

		BlockIndicator[] pmmlBlockIndicators = null;

		if(blockIndicators != null){
			List<Feature> blockIndicatorFeatures = BlockIndicatorUtil.selectFeatures(blockIndicators, features);

			features = new ArrayList<>(features);
			features.removeAll(blockIndicatorFeatures);

			pmmlBlockIndicators = BlockIndicatorUtil.toBlockIndicators(blockIndicatorFeatures);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			Field<?> field = feature.getField();

			Lag lag = new Lag(field.requireName())
				.setAggregate(aggregate)
				.setN(n);

			if(pmmlBlockIndicators != null){
				lag = lag.addBlockIndicators(pmmlBlockIndicators);
			}

			DerivedField derivedField = encoder.createDerivedField(FieldNameUtil.create(aggregate.value(), feature, n), OpType.CONTINUOUS, DataType.DOUBLE, lag);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public String getFunction(){
		return getEnum("function", this::getString, Arrays.asList(RollingAggregateTransformer.FUNCTION_AVG, RollingAggregateTransformer.FUNCTION_MAX, RollingAggregateTransformer.FUNCTION_MEAN, RollingAggregateTransformer.FUNCTION_MIN, RollingAggregateTransformer.FUNCTION_PROD, RollingAggregateTransformer.FUNCTION_PRODUCT, RollingAggregateTransformer.FUNCTION_SUM));
	}

	public Integer getN(){
		return getInteger("n");
	}

	public List<Object> getBlockIndicators(){

		if(!hasattr("block_indicators")){
			return null;
		}

		return getObjectList("block_indicators");
	}

	static
	private Lag.Aggregate parseFunction(String function){

		switch(function){
			case RollingAggregateTransformer.FUNCTION_AVG: // PMML-style
			case RollingAggregateTransformer.FUNCTION_MEAN: // Python-style
				return Lag.Aggregate.AVG;
			case RollingAggregateTransformer.FUNCTION_MAX:
				return Lag.Aggregate.MAX;
			case RollingAggregateTransformer.FUNCTION_MIN:
				return Lag.Aggregate.MIN;
			case RollingAggregateTransformer.FUNCTION_PROD: // Python-style
			case RollingAggregateTransformer.FUNCTION_PRODUCT: // PMML-style
				return Lag.Aggregate.PRODUCT;
			case RollingAggregateTransformer.FUNCTION_SUM:
				return Lag.Aggregate.SUM;
			default:
				throw new IllegalArgumentException(function);
		}
	}

	private static final String FUNCTION_AVG = "avg";
	private static final String FUNCTION_MAX = "max";
	private static final String FUNCTION_MEAN = "mean";
	private static final String FUNCTION_MIN = "min";
	private static final String FUNCTION_PROD = "prod";
	private static final String FUNCTION_PRODUCT = "product";
	private static final String FUNCTION_SUM = "sum";
}