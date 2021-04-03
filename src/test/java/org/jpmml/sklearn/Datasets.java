/*
 * Copyright (c) 2021 Villu Ruusmann
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

import org.dmg.pmml.FieldName;
import org.jpmml.converter.FieldNameUtil;

interface Datasets {

	String APOLLO = "Apollo";
	String AUDIT = "Audit";
	String AUDIT_CAT = AUDIT + "Cat";
	String AUDIT_DICT = AUDIT + "Dict";
	String AUDIT_NA = AUDIT + "NA";
	String AUTO = "Auto";
	String AUTO_NA = AUTO + "NA";
	String HOUSING = "Housing";
	String IRIS = "Iris";
	String SENTIMENT = "Sentiment";
	String VERSICOLOR = "Versicolor";
	String VISIT = "Visit";
	String WHEAT = "Wheat";

	FieldName AUDIT_ADJUSTED = FieldName.create("Adjusted");
	FieldName AUDIT_PROBABILITY_TRUE = FieldNameUtil.create("probability", 1);
	FieldName AUDIT_PROBABILITY_FALSE = FieldNameUtil.create("probability", 0);

	FieldName AUTO_MPG = FieldName.create("mpg");

	FieldName IRIS_SPECIES = FieldName.create("Species");
	FieldName IRIS_PROBABILITY_SETOSA = FieldNameUtil.create("probability", "setosa");
	FieldName IRIS_PROBABILITY_VERSICOLOR = FieldNameUtil.create("probability", "versicolor");
	FieldName IRIS_PROBABILITY_VIRGINICA = FieldNameUtil.create("probability", "virginica");
}