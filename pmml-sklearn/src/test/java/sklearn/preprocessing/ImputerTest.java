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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import numpy.core.NDArrayUtil;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Decorator;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import org.junit.jupiter.api.Test;
import sklearn.Transformer;
import sklearn2pmml.decoration.CategoricalDomain;
import sklearn2pmml.decoration.ContinuousDomain;
import sklearn_pandas.DataFrameMapper;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ImputerTest {

	@Test
	public void encodeCategorical(){
		String name = "x";
		String imputedName = FieldNameUtil.create("imputer", name);

		Imputer imputer = new Imputer("sklearn.preprocessing.imputation", "Imputer");
		imputer.put("strategy", "most_frequent");
		imputer.put("missing_values", "NaN");
		imputer.put("statistics_", 0);

		SkLearnEncoder encoder = new SkLearnEncoder();

		Feature feature = encodeFeature(name, Arrays.asList(imputer), encoder);

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		List<Decorator> decorators = getDecorators(encoder, name);

		assertEquals(1, decorators.size());

		assertTrue(feature instanceof WildcardFeature);
		assertEquals(name, feature.getName());

		CategoricalDomain categoricalDomain = new CategoricalDomain("sklearn2pmml.decoration", "CategoricalDomain");
		categoricalDomain.put("invalid_value_treatment", "as_is");
		categoricalDomain.put("data_", NDArrayUtil.toArray(Arrays.asList(0, 1, 2, 3, 4, 5, 6)));

		encoder = new SkLearnEncoder();

		feature = encodeFeature(name, Arrays.asList(categoricalDomain, imputer), encoder);

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		decorators = getDecorators(encoder, name);

		assertEquals(2, decorators.size());

		assertTrue(feature instanceof CategoricalFeature);
		assertEquals(name, feature.getName());
	}

	@Test
	public void encodeContinuous(){
		String name = "x";
		String imputedName = FieldNameUtil.create("imputer", name);
		String binarizedName = FieldNameUtil.create("binarizer", name);
		String imputedBinarizedName = FieldNameUtil.create("imputer", binarizedName);

		Imputer imputer = new Imputer("sklearn.preprocessing.imputation", "Imputer");
		imputer.put("strategy", "mean");
		imputer.put("missing_values", -999);
		imputer.put("statistics_", 0.5d);

		SkLearnEncoder encoder = new SkLearnEncoder();

		Feature feature = encodeFeature(name, Arrays.asList(imputer), encoder);

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		List<Decorator> decorators = getDecorators(encoder, name);

		assertEquals(1, decorators.size());

		assertTrue(feature instanceof WildcardFeature);
		assertEquals(name, feature.getName());

		ContinuousDomain continuousDomain = new ContinuousDomain("sklearn2pmml.decoration", "ContinuousDomain");
		continuousDomain.put("invalid_value_treatment", "return_invalid");
		continuousDomain.put("data_min_", 0d);
		continuousDomain.put("data_max_", 1d);

		encoder = new SkLearnEncoder();

		feature = encodeFeature(name, Arrays.asList(continuousDomain, imputer), encoder);

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		decorators = getDecorators(encoder, name);

		assertEquals(2, decorators.size());

		assertTrue(feature instanceof ContinuousFeature);
		assertEquals(name, feature.getName());

		Binarizer binarizer = new Binarizer("sklearn.preprocessing.data", "Binarizer");
		binarizer.put("threshold", 1d / 3d);

		encoder = new SkLearnEncoder();

		feature = encodeFeature(name, Arrays.asList(continuousDomain, binarizer, imputer), encoder);

		assertNotNull(encoder.getDataField(name));
		assertNotNull(encoder.getDerivedField(binarizedName));
		assertNotNull(encoder.getDerivedField(imputedBinarizedName));

		decorators = getDecorators(encoder, name);

		assertEquals(1, decorators.size());

		assertTrue(feature instanceof ContinuousFeature);
		assertEquals(imputedBinarizedName, feature.getName());
	}

	static
	private List<Decorator> getDecorators(ModelEncoder encoder, String name){
		return (encoder.getDecorators()).get(null).get(name);
	}

	static
	private Feature encodeFeature(String name, List<? extends Transformer> transformers, SkLearnEncoder encoder){
		DataFrameMapper dataFrameMapper = new DataFrameMapper("sklearn_pandas.dataframe_mapper", "DataFrameMapper")
			.setDefault(Boolean.FALSE)
			.setFeatures(Collections.singletonList(new Object[]{name, transformers}));

		List<Feature> features = dataFrameMapper.encodeFeatures(new ArrayList<Feature>(), encoder);

		return Iterables.getOnlyElement(features);
	}
}