/*
 * Copyright (c) 2016 Villu Ruusmann
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

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Value;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PseudoFeature;
import org.jpmml.sklearn.FeatureMapper;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class OneHotEncoderTest {

	@Test
	public void encode(){
		FeatureMapper featureMapper = new FeatureMapper();

		DataField dataField = featureMapper.createDataField(FieldName.create("x"), OpType.CATEGORICAL, DataType.INTEGER);

		Feature inputFeature = new PseudoFeature(dataField);

		assertEquals(Arrays.asList(), getValues(dataField));

		OneHotEncoder encoder = new OneHotEncoder("sklearn.preprocessing.data", "OneHotEncoder");
		encoder.put("n_values_", 3);

		List<Feature> inputFeatures = Collections.singletonList(inputFeature);
		List<Feature> outputFeatures = encoder.encodeFeatures(Collections.singletonList("x"), inputFeatures, featureMapper);

		for(int i = 0; i < 3; i++){
			BinaryFeature outputFeature = (BinaryFeature)outputFeatures.get(i);

			assertEquals(String.valueOf(i), outputFeature.getValue());
		}

		assertEquals(Arrays.asList("0", "1", "2"), getValues(dataField));
	}

	static
	private List<String> getValues(DataField dataField){
		List<Value> values = dataField.getValues();

		Function<Value, String> function = new Function<Value, String>(){

			@Override
			public String apply(Value value){
				return value.getValue();
			}
		};

		return Lists.transform(values, function);
	}
}