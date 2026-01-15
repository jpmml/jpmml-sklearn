/*
 * Copyright (c) 2026 Villu Ruusmann
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
package org.jpmml.sklearn.xgboost.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class FLAMLTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Test
	public void evaluateXGBoostLimitDepthEstimatorAudit() throws Exception {
		evaluate("FLAMLLimitDepth", AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(8 + 4));
	}

	@Test
	public void evaluateXGBoostSklearnEstimatorAudit() throws Exception {
		evaluate("FLAML", AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(8 + 4));
	}

	@Test
	public void evaluateXGBoostLimitDepthEstimatorAuto() throws Exception {
		evaluate("FLAMLLimitDepth", AUTO, new FloatEquivalence(4 + 4));
	}

	@Test
	public void evaluateXGBoostSklearnEstimatorAuto() throws Exception {
		evaluate("FLAML", AUTO, new FloatEquivalence(4));
	}
}