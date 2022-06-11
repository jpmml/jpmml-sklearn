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
package org.jpmml.sklearn.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.junit.Test;

public class SkLearn2PMMLTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Test
	public void evaluateCHAIDAudit() throws Exception {
		evaluate("CHAID", AUDIT, excludeFields(AUDIT_ADJUSTED, AUDIT_PROBABILITY_FALSE, AUDIT_PROBABILITY_TRUE));
	}

	@Test
	public void evaluateCHAIDAuditNA() throws Exception {
		evaluate("CHAID", AUDIT_NA, excludeFields(AUDIT_ADJUSTED, AUDIT_PROBABILITY_FALSE, AUDIT_PROBABILITY_TRUE));
	}

	@Test
	public void evaluateCHAIDAuto() throws Exception {
		evaluate("CHAID", AUTO, excludeFields(AUTO_MPG));
	}

	@Test
	public void evaluateCHAIDAutoNA() throws Exception {
		evaluate("CHAID", AUTO_NA, excludeFields(AUTO_MPG));
	}

	@Test
	public void evaluateCHAIDIris() throws Exception {
		evaluate("CHAID", IRIS, excludeFields(IRIS_SPECIES, IRIS_PROBABILITY_SETOSA, IRIS_PROBABILITY_VERSICOLOR, IRIS_PROBABILITY_VIRGINICA));
	}

	@Test
	public void evaluateMLPAutoencoderIris() throws Exception {
		evaluate("MLPAutoencoder", IRIS);
	}

	@Test
	public void evaluateMLPTransformerIris() throws Exception {
		evaluate("MLPTransformer", IRIS);
	}
}