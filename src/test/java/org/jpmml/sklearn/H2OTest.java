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

import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Fields;
import org.junit.Test;

public class H2OTest extends SkLearnTest implements SkLearnAlgorithms, SkLearnDatasets, Fields {

	@Test
	public void evaluateGradientBoostingAudit() throws Exception {
		String[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2O" + GRADIENT_BOOSTING, AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateGradientBoostingAuto() throws Exception {
		evaluate("H2O" + GRADIENT_BOOSTING, AUTO);
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		String[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2O" + LOGISTIC_REGRESSION, AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("H2O" + LINEAR_REGRESSION, AUTO);
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		String[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2O" + RANDOM_FOREST, AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate("H2O" + RANDOM_FOREST, AUTO);
	}

	static
	private String[] createTargetFields(String name){
		return new String[]{name, FieldNameUtil.create("h2o", name)};
	}
}