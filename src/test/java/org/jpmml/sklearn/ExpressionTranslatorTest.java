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
package org.jpmml.sklearn;

import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.junit.Test;

public class ExpressionTranslatorTest {

	@Test
	public void translate(){
		List<Feature> features = Arrays.<Feature>asList(
			new ContinuousFeature(null, FieldName.create("a"), DataType.DOUBLE),
			new ContinuousFeature(null, FieldName.create("b"), DataType.DOUBLE),
			new ContinuousFeature(null, FieldName.create("c"), DataType.DOUBLE)
		);

		ExpressionTranslator.translate("(X[:, 0] + X[:, 1] - 1) / X[:, 2] * -0.5", features);
	}
}