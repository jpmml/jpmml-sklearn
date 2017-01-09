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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import numpy.core.NDArray;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldDecorator;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import org.junit.Test;
import sklearn.Transformer;
import sklearn2pmml.decoration.CategoricalDomain;
import sklearn2pmml.decoration.ContinuousDomain;
import sklearn_pandas.DataFrameMapper;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public class ImputerTest {

	@Test
	public void encodeCategorical(){
		FieldName name = FieldName.create("x");
		FieldName imputedName = FieldName.create("imputer(x)");

		Imputer imputer = new Imputer("sklearn.preprocessing.imputation", "Imputer");
		imputer.put("strategy", "most_frequent");
		imputer.put("missing_values", "N/A");
		imputer.put("statistics_", 0);

		SkLearnEncoder encoder = encode(name.getValue(), Arrays.asList(imputer));

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		List<FieldDecorator> decorators = encoder.getDecorators(name);

		assertEquals(1, decorators.size());

		Feature feature = Iterables.getOnlyElement(encoder.getFeatures());

		assertTrue(feature instanceof WildcardFeature);
		assertEquals(name, feature.getName());

		NDArray array = new NDArray();
		array.put("data", Arrays.asList(0, 1, 2, 3, 4, 5, 6));
		array.put("fortran_order", Boolean.FALSE);

		CategoricalDomain categoricalDomain = new CategoricalDomain("sklearn2pmml.decoration", "CategoricalDomain");
		categoricalDomain.put("invalid_value_treatment", "as_is");
		categoricalDomain.put("data_", array);

		encoder = encode(name.getValue(), Arrays.asList(categoricalDomain, imputer));

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		decorators = encoder.getDecorators(name);

		assertEquals(2, decorators.size());

		feature = Iterables.getOnlyElement(encoder.getFeatures());

		assertTrue(feature instanceof CategoricalFeature);
		assertEquals(name, feature.getName());
	}

	@Test
	public void encodeContinuous(){
		FieldName name = FieldName.create("x");
		FieldName imputedName = FieldName.create("imputer(x)");

		Imputer imputer = new Imputer("sklearn.preprocessing.imputation", "Imputer");
		imputer.put("strategy", "mean");
		imputer.put("missing_values", -999);
		imputer.put("statistics_", 0.5d);

		SkLearnEncoder encoder = encode(name.getValue(), Arrays.asList(imputer));

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		List<FieldDecorator> decorators = encoder.getDecorators(name);

		assertEquals(1, decorators.size());

		Feature feature = Iterables.getOnlyElement(encoder.getFeatures());

		assertTrue(feature instanceof WildcardFeature);
		assertEquals(name, feature.getName());

		ContinuousDomain continuousDomain = new ContinuousDomain("sklearn2pmml.decoration", "ContinuousDomain");
		continuousDomain.put("invalid_value_treatment", "return_invalid");
		continuousDomain.put("data_min_", 0d);
		continuousDomain.put("data_max_", 1d);

		encoder = encode(name.getValue(), Arrays.asList(continuousDomain, imputer));

		assertNotNull(encoder.getDataField(name));
		assertNull(encoder.getDerivedField(imputedName));

		decorators = encoder.getDecorators(name);

		assertEquals(2, decorators.size());

		feature = Iterables.getOnlyElement(encoder.getFeatures());

		assertTrue(feature instanceof ContinuousFeature);
		assertEquals(name, feature.getName());

		Binarizer binarizer = new Binarizer("sklearn.preprocessing.data", "Binarizer");
		binarizer.put("threshold", 1d / 3d);

		encoder = encode(name.getValue(), Arrays.asList(continuousDomain, binarizer, imputer));

		assertNotNull(encoder.getDataField(name));
		assertNotNull(encoder.getDerivedField(FieldName.create("binarizer(x)")));
		assertNotNull(encoder.getDerivedField(imputedName));

		decorators = encoder.getDecorators(name);

		assertEquals(1, decorators.size());

		feature = Iterables.getOnlyElement(encoder.getFeatures());

		assertTrue(feature instanceof ContinuousFeature);
		assertEquals(imputedName, feature.getName());
	}

	static
	private SkLearnEncoder encode(String name, List<? extends Transformer> transformers){
		return encode(Collections.singletonList(new Object[]{name, transformers}));
	}

	static
	private SkLearnEncoder encode(List<Object[]> features){
		SkLearnEncoder encoder = new SkLearnEncoder();

		DataFrameMapper dataFrameMapper = new DataFrameMapper("sklearn_pandas.dataframe_mapper", "DataFrameMapper");
		dataFrameMapper.put("features", features);

		dataFrameMapper.encodeFeatures(encoder);

		return encoder;
	}
}