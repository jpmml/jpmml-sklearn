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
import org.jpmml.converter.FieldNameUtil;
import org.junit.Test;

public class BaseEstimatorTest extends SkLearnTest implements Datasets {

	@Test
	public void evaluateGradientBoostingAudit() throws Exception {
		FieldName[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2OGradientBoosting", AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateGradientBoostingAuto() throws Exception {
		evaluate("H2OGradientBoosting", AUTO);
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		FieldName[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2OLogisticRegression", AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("H2OLinearRegression", AUTO);
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		FieldName[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2ORandomForest", AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate("H2ORandomForest", AUTO);
	}

	static
	private FieldName[] createTargetFields(FieldName name){
		return new FieldName[]{name, FieldNameUtil.create("h2o", name)};
	}
}