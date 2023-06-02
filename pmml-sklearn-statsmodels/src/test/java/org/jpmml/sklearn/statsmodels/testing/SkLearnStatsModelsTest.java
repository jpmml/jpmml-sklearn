/*
 * Copyright (c) 2022 Villu Ruusmann
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
package org.jpmml.sklearn.statsmodels.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;

public class SkLearnStatsModelsTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluateGLMAudit() throws Exception {
		evaluate("GLM", AUDIT);
	}

	@Test
	public void evaluateLogitAudit() throws Exception {
		evaluate("Logit", AUDIT);
	}

	@Test
	public void evaluateGLMAuto() throws Exception {
		evaluate("GLM", AUTO);
	}

	@Test
	public void evaluateOLSAuto() throws Exception {
		evaluate("OLS", AUTO);
	}

	@Test
	public void evaluateOrderedLogitAuto() throws Exception {
		evaluate("OrderedLogit", AUTO);
	}

	@Test
	public void evaluateWLSAuto() throws Exception {
		evaluate("WLS", AUTO);
	}

	@Test
	public void evaluateMNLogitIris() throws Exception {
		evaluate("MNLogit", IRIS);
	}

	@Test
	public void evaluateGLMVisit() throws Exception {
		evaluate("GLM", VISIT);
	}

	@Test
	public void evaluatePoissonVisit() throws Exception {
		evaluate("Poisson", VISIT);
	}
}