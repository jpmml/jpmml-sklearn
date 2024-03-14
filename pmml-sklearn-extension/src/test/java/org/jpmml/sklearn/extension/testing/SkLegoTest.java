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
package org.jpmml.sklearn.extension.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;

public class SkLegoTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluateEstimatorTransformerAudit() throws Exception {
		evaluate("EstimatorTransformer", AUDIT);
	}

	@Test
	public void evaluateEstimatorTransformerAuto() throws Exception {
		evaluate("EstimatorTransformer", AUTO);
	}

	@Test
	public void evaluateEstimatorTransformerHousing() throws Exception {
		evaluate("EstimatorTransformer", HOUSING);
	}

	@Test
	public void evaluateEstimatorTransformerIris() throws Exception {
		evaluate("EstimatorTransformer", IRIS);
	}

	@Test
	public void evaluateEstimatorTransformerVersicolor() throws Exception {
		evaluate("EstimatorTransformer", VERSICOLOR);
	}

	@Test
	public void evaluateEstimatorTransformerVisit() throws Exception {
		evaluate("EstimatorTransformer", VISIT);
	}

	@Test
	public void evaluateEstimatorTransformerWheat() throws Exception {
		evaluate("EstimatorTransformer", WHEAT);
	}

	@Test
	public void evaluatePrdinalClassifierAuto() throws Exception {
		evaluate("OrdinalClassifier", AUTO, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateOrdinalClassifierIsotonicAuto() throws Exception {
		evaluate("OrdinalClassifierIsotonic", AUTO, new PMMLEquivalence(1e-12, 1e-12));
	}

	@Test
	public void evaluateOrdinalClassifierSigmoidAuto() throws Exception {
		evaluate("OrdinalClassifierSigmoid", AUTO);
	}
}