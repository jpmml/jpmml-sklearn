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
import org.jpmml.sklearn.testing.SkLearnAlgorithms;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class FLAMLTest extends SkLearnEncoderBatchTest implements Datasets, SkLearnAlgorithms {

	@Test
	public void evaluateExtraTreesEstimatorAudit() throws Exception {
		evaluate(EXTRA_TREES + "Estimator", AUDIT);
	}

	@Test
	public void evaluateHistGradientBoostingEstimatorAudit() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING + "Estimator", AUDIT);
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
		evaluate(RANDOM_FOREST + "Estimator", AUDIT);
	}

	@Test
	public void evaluateSGDEstimatorAudit() throws Exception {
		evaluate(SGD + "Estimator", AUDIT);
	}

	@Test
	public void evaluateSVCEstimatorAudit() throws Exception {
		evaluate(SVC + "Estimator", AUDIT);
	}

	@Test
	public void evaluateExtraTreesEstimatorAuto() throws Exception {
		evaluate(EXTRA_TREES + "Estimator", AUTO);
	}

	@Test
	public void evaluateElasticNetEstimatorAuto() throws Exception {
		evaluate(ELASTIC_NET + "Estimator", AUTO);
	}

	@Test
	public void evaluateHistGradientBoostingEstimatorAuto() throws Exception {
		evaluate(HIST_GRADIENT_BOOSTING + "Estimator", AUTO);
	}

	@Test
	public void evaluateLassoLarsEstimatorAuto() throws Exception {
		evaluate(LASSO_LARS + "Estimator", AUTO);
	}

	@Test
	public void evaluateRandomForestEstimatorAuto() throws Exception {
		evaluate(RANDOM_FOREST + "Estimator", AUTO);
	}

	@Test
	public void evaluateSGDEstimatorAuto() throws Exception {
		evaluate(SGD + "Estimator", AUTO);
	}
}