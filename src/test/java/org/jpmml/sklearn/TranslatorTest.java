/*
 * Copyright (c) 2018 Villu Ruusmann
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
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ObjectFeature;

abstract
class TranslatorTest {

	static final List<Feature> booleanFeatures = Arrays.asList(
		new CategoricalFeature(null, FieldName.create("a"), DataType.BOOLEAN, Arrays.asList("false", "true")),
		new CategoricalFeature(null, FieldName.create("b"), DataType.BOOLEAN, Arrays.asList("false", "true")),
		new CategoricalFeature(null, FieldName.create("c"), DataType.BOOLEAN, Arrays.asList("false", "true"))
	);

	static final List<Feature> doubleFeatures = Arrays.asList(
		new ContinuousFeature(null, FieldName.create("a"), DataType.DOUBLE),
		new ContinuousFeature(null, FieldName.create("b"), DataType.DOUBLE),
		new ContinuousFeature(null, FieldName.create("c"), DataType.DOUBLE)
	);

	static final List<Feature> stringFeatures = Arrays.asList(
		new ObjectFeature(null, FieldName.create("a"), DataType.STRING),
		new ObjectFeature(null, FieldName.create("b"), DataType.STRING),
		new ObjectFeature(null, FieldName.create("c"), DataType.STRING)
	);
}