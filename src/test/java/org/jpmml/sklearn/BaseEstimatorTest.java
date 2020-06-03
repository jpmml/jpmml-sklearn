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

import org.dmg.pmml.FieldName;
import org.junit.Test;

public class BaseEstimatorTest extends SkLearnTest {

	@Test
	public void evaluateGradientBoostingAudit() throws Exception {
		FieldName[] targetFields = {FieldName.create("Adjusted"), FieldName.create("h2o(Adjusted)")};

		evaluate("H2OGradientBoosting", "Audit", excludeFields(targetFields));
	}

	@Test
	public void evaluateGradientBoostingAuto() throws Exception {
		evaluate("H2OGradientBoosting", "Auto");
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		FieldName[] targetFields = {FieldName.create("Adjusted"), FieldName.create("h2o(Adjusted)")};

		evaluate("H2OLogisticRegression", "Audit", excludeFields(targetFields));
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("H2OLinearRegression", "Auto");
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		FieldName[] targetFields = {FieldName.create("Adjusted"), FieldName.create("h2o(Adjusted)")};

		evaluate("H2ORandomForest", "Audit", excludeFields(targetFields));
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate("H2ORandomForest", "Auto");
	}
}