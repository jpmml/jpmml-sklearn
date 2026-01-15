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
package org.jpmml.sklearn.extension.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class FLAMLTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluateExtraTreesEstimatorAudit() throws Exception {
		evaluate("ExtraTreesEstimator", AUDIT);
	}

	@Test
	public void evaluateHistGradientBoostingEstimatorAudit() throws Exception {
		evaluate("HistGradientBoostingEstimator", AUDIT);
	}

	@Test
	public void evaluateLGBMEstimatorAudit() throws Exception {
		evaluate("LGBMEstimator", AUDIT);
	}

	@Test
	public void evaluateLRL1ClassifierAudit() throws Exception {
		evaluate("LRL1Classifier", AUDIT);
	}

	@Test
	public void evaluateLRL2ClassifierAudit() throws Exception {
		evaluate("LRL2Classifier", AUDIT);
	}

	@Test
	public void evaluateRandomForestEstimatorAudit() throws Exception {
		evaluate("RandomForestEstimator", AUDIT);
	}

	@Test
	public void evaluateSVCEstimatorAudit() throws Exception {
		evaluate("SVCEstimator", AUDIT);
	}

	@Test
	public void evaluateXGBoostLimitDepthEstimatorAudit() throws Exception {
		evaluate("XGBoostLimitDepthEstimator", AUDIT, new PMMLEquivalence(5e-6, 5e-6));
	}

	@Test
	public void evaluateXGBoostSklearnEstimatorAudit() throws Exception {
		evaluate("XGBoostSklearnEstimator", AUDIT, new PMMLEquivalence(1e-6, 1e-6));
	}

	@Test
	public void evaluateExtraTreesEstimatorAuto() throws Exception {
		evaluate("ExtraTreesEstimator", AUTO);
	}

	@Test
	public void evaluateElasticNetEstimatorAuto() throws Exception {
		evaluate("ElasticNetEstimator", AUTO);
	}

	@Test
	public void evaluateHistGradientBoostingEstimatorAuto() throws Exception {
		evaluate("HistGradientBoostingEstimator", AUTO);
	}

	@Test
	public void evaluateLassoLarsEstimatorAuto() throws Exception {
		evaluate("LassoLarsEstimator", AUTO);
	}

	@Test
	public void evaluateLGBMEstimatorAuto() throws Exception {
		evaluate("LGBMEstimator", AUTO);
	}

	@Test
	public void evaluateRandomForestEstimatorAuto() throws Exception {
		evaluate("RandomForestEstimator", AUTO);
	}

	@Test
	public void evaluateXGBoostLimitDepthEstimatorAuto() throws Exception {
		evaluate("XGBoostLimitDepthEstimator", AUTO, new PMMLEquivalence(1e-6, 1e-6));
	}

	@Test
	public void evaluateXGBoostSklearnEstimatorAuto() throws Exception {
		evaluate("XGBoostSklearnEstimator", AUTO, new PMMLEquivalence(1e-6, 1e-6));
	}
}