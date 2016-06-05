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

import java.util.Collections;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PseudoFeature;
import org.jpmml.sklearn.FeatureMapper;
import sklearn.Transformer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

abstract
class ScalerTest {

	void assertSameFeature(Transformer transformer){
		FeatureMapper featureMapper = new FeatureMapper();

		DataField dataField = featureMapper.createDataField(FieldName.create("x"));

		Feature inputFeature = new PseudoFeature(dataField);
		Feature outputFeature = Iterables.getOnlyElement(transformer.encodeFeatures(Collections.singletonList("x"), Collections.singletonList(inputFeature), featureMapper));

		assertSame(inputFeature, outputFeature);
	}

	void assertTransformedFeature(Transformer transformer, String function){
		FeatureMapper featureMapper = new FeatureMapper();

		DataField dataField = featureMapper.createDataField(FieldName.create("x"));

		Feature inputFeature = new PseudoFeature(dataField);
		Feature outputFeature = Iterables.getOnlyElement(transformer.encodeFeatures(Collections.singletonList("x"), Collections.singletonList(inputFeature), featureMapper));

		assertNotSame(inputFeature, outputFeature);

		DerivedField derivedField = (DerivedField)featureMapper.getField(outputFeature.getName());

		Apply apply = (Apply)derivedField.getExpression();

		assertEquals(function, apply.getFunction());
	}
}