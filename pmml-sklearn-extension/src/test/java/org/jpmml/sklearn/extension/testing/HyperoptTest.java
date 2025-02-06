/*
 * Copyright (c) 2023 Villu Ruusmann
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
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class HyperoptTest extends SkLearnEncoderBatchTest implements Datasets {

	@Test
	public void evaluateHyperoptAudit() throws Exception {
		evaluate("Hyperopt", AUDIT);
	}

	@Test
	public void evaluateHyperoptAuto() throws Exception {
		evaluate("Hyperopt", AUTO);
	}

	@Test
	public void evaluateHyperoptIris() throws Exception {
		evaluate("Hyperopt", IRIS);
	}
}