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

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class OneHotEncoderTest {

	@Test
	public void encode(){
		SkLearnEncoder encoder = new SkLearnEncoder();

		DataField dataField = encoder.createDataField(FieldName.create("x"), OpType.CATEGORICAL, DataType.INTEGER);

		Feature inputFeature = new WildcardFeature(encoder, dataField);

		assertEquals(Arrays.asList(), PMMLUtil.getValues(dataField));

		OneHotEncoder oneHotEncoder = new OneHotEncoder("sklearn.preprocessing.data", "OneHotEncoder");
		oneHotEncoder.put("n_values_", 3);

		List<Feature> outputFeatures = oneHotEncoder.encodeFeatures(Collections.singletonList(inputFeature), encoder);
		for(int i = 0; i < 3; i++){
			BinaryFeature outputFeature = (BinaryFeature)outputFeatures.get(i);

			assertEquals(String.valueOf(i), outputFeature.getValue());
		}

		assertEquals(Arrays.asList("0", "1", "2"), PMMLUtil.getValues(dataField));
	}
}