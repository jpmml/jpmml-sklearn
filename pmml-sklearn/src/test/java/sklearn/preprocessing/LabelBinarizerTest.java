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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import numpy.core.NDArrayUtil;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.SkLearnEncoder;
import org.junit.jupiter.api.Test;
import sklearn.SkLearnFields;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LabelBinarizerTest {

	@Test
	public void encode(){
		SkLearnEncoder encoder = new SkLearnEncoder();

		DataField dataField = encoder.createDataField("x", OpType.CATEGORICAL, DataType.STRING);

		Feature inputFeature = new WildcardFeature(encoder, dataField);

		LabelBinarizer binarizer = new LabelBinarizer("sklearn.preprocessing.label", "LabelBinarizer");
		binarizer.put(SkLearnFields.CLASSES, NDArrayUtil.toArray(Arrays.asList("low", "medium", "high")));
		binarizer.put("pos_label", 1d);
		binarizer.put("neg_label", -1d);

		List<Feature> outputFeatures = binarizer.encode(Collections.singletonList(inputFeature), encoder);
		for(Feature outputFeature : outputFeatures){
			assertTrue(outputFeature instanceof CategoricalFeature);
		}

		assertEquals(Arrays.asList("low", "medium", "high"), FieldUtil.getValues(dataField));

		binarizer.put("neg_label", 0d);

		outputFeatures = binarizer.encode(Collections.singletonList(inputFeature), encoder);
		for(Feature outputFeature : outputFeatures){
			assertTrue(outputFeature instanceof BinaryFeature);
		}

		assertEquals(Arrays.asList("low", "medium", "high"), FieldUtil.getValues(dataField));
	}
}