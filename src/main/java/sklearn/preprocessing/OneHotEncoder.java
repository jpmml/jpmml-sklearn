/*
 * Copyright (c) 2015 Villu Ruusmann
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

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.ContiguousSet;
import com.google.common.collect.DiscreteDomain;
import com.google.common.collect.Range;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class OneHotEncoder extends Transformer {

	public OneHotEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		List<? extends Number> values = getValues();

		return TypeUtil.getDataType(values, DataType.INTEGER);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<? extends Number> values = getValues();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		List<Feature> result = new ArrayList<>();

		if(feature instanceof CategoricalFeature){
			CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

			ClassDictUtil.checkSize(values, categoricalFeature.getValues());

			for(int i = 0; i < values.size(); i++){
				result.add(new BinaryFeature(encoder, categoricalFeature, categoricalFeature.getValue(i)));
			}
		} else

		if(feature instanceof WildcardFeature){
			WildcardFeature wildcardFeature = (WildcardFeature)feature;

			List<Integer> categories = new ArrayList<>();

			for(int i = 0; i < values.size(); i++){
				Number value = values.get(i);

				Integer category = ValueUtil.asInt(value);

				categories.add(category);

				result.add(new BinaryFeature(encoder, wildcardFeature, category));
			}

			wildcardFeature.toCategoricalFeature(categories);
		} else

		{
			throw new IllegalArgumentException();
		}

		return result;
	}

	public List<? extends Number> getValues(){
		List<Integer> featureSizes = getFeatureSizes();

		ClassDictUtil.checkSize(1, featureSizes);

		Object numberOfValues = getOptionalScalar("n_values");

		if(("auto").equals(numberOfValues)){
			return getActiveFeatures();
		}

		Integer featureSize = featureSizes.get(0);

		List<Integer> result = new ArrayList<>();
		result.addAll(ContiguousSet.create(Range.closedOpen(0, featureSize), DiscreteDomain.integers()));

		return result;
	}

	public List<? extends Number> getActiveFeatures(){
		return getNumberArray("active_features_");
	}

	public List<Integer> getFeatureSizes(){
		return getIntegerArray("n_values_");
	}
}