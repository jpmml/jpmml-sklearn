/*
 * Copyright (c) 2024 Villu Ruusmann
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
package org.jpmml.sklearn.extension.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class InterpretTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluateExplainableBoostingClassifierAudit() throws Exception {
		evaluate("ExplainableBoostingClassifier", AUDIT, excludeFields("ebm(Age)", "ebm(Education)"));
	}

	@Test
	public void evaluateExplainableBoostingClassifierAuditNA() throws Exception {
		evaluate("ExplainableBoostingClassifier", AUDIT_NA);
	}

	@Test
	public void evaluateExplainableBoostingRegressorAuto() throws Exception {
		evaluate("ExplainableBoostingRegressor", AUTO, excludeFields("ebm(acceleration)", "ebm(cylinders)"));
	}

	@Test
	public void evaluateExplainableBoostingRegressorAutoNA() throws Exception {
		evaluate("ExplainableBoostingRegressor", AUTO_NA);
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("LinearRegression", AUTO);
	}

	@Test
	public void evaluateRegressionTreeAuto() throws Exception {
		evaluate("RegressionTree", AUTO);
	}

	@Test
	public void evaluateClassificationTreeIris() throws Exception {
		evaluate("ClassificationTree", IRIS);
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate("LogisticRegression", IRIS);
	}

	@Test
	public void evaluateExplainableBoostingClassifierVersicolor() throws Exception {
		evaluate("ExplainableBoostingClassifier", VERSICOLOR, new PMMLEquivalence(5e-11, 5e-11));
	}
}