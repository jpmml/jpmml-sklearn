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

import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.junit.Test;

public class TPOTTest extends SkLearnEncoderBatchTest implements SkLearnDatasets {

	@Test
	public void evaluateTPOTAudit() throws Exception {
		evaluate("TPOT", AUDIT);
	}

	@Test
	public void evaluateTPOTAuto() throws Exception {
		evaluate("TPOT", AUTO);
	}

	@Test
	public void evaluateTPOTHousing() throws Exception {
		evaluate("TPOT", HOUSING);
	}

	@Test
	public void evaluateTPOTIris() throws Exception {
		evaluate("TPOT", IRIS);
	}

	@Test
	public void evaluateTPOTVersicolor() throws Exception {
		evaluate("TPOT", VERSICOLOR, new PMMLEquivalence(5e-13, 5e-13));
	}
}