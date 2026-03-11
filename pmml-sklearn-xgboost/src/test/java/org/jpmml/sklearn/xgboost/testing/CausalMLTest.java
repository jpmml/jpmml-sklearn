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

import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class CausalMLTest extends SkLearnEncoderBatchTest {

	public CausalMLTest(){
		super(new PMMLEquivalence(1e-7, 1e-7));
	}

	@Test
	public void evaluateXGBRRegressorEmail() throws Exception {
		evaluate("XGBRRegressor", "Email");
	}

	@Test
	public void evaluateXGBTRegressorEmail() throws Exception {
		evaluate("XGBTRegressor", "Email", new PMMLEquivalence(3e-4, 3e-4));
	}
}