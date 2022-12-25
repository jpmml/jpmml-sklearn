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
package org.jpmml.sklearn.extension.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;

public class OptBinningTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Test
	public void evaluateBinningProcessAudit() throws Exception {
		evaluate("BinningProcess", AUDIT);
	}

	@Test
	public void evaluateBinningProcessAuditNA() throws Exception {
		evaluate("BinningProcess", AUDIT_NA);
	}

	@Test
	public void evaluateOptimalBinningAudit() throws Exception {
		evaluate("OptimalBinning", AUDIT);
	}

	@Test
	public void evaluateOptimalBinningAuditNA() throws Exception {
		evaluate("OptimalBinning", AUDIT_NA);
	}

	@Test
	public void evaluateScorecardAudit() throws Exception {
		evaluate("Scorecard", AUDIT);
	}

	@Test
	public void evaluateScaledScorecardAudit() throws Exception {
		evaluate("ScaledScorecard", AUDIT);
	}

	@Test
	public void evaluateBinningProcessAuto() throws Exception {
		evaluate("BinningProcess", AUTO);
	}

	@Test
	public void evaluateBinningProcessAutoNA() throws Exception {
		evaluate("BinningProcess", AUTO_NA);
	}

	@Test
	public void evaluateScorecardAuto() throws Exception {
		evaluate("Scorecard", AUTO);
	}

	@Test
	public void evaluateScorecardAutoNA() throws Exception {
		evaluate("Scorecard", AUTO_NA);
	}

	@Test
	public void evaluateScaledScorecardAuto() throws Exception {
		evaluate("ScaledScorecard", AUTO);
	}

	@Test
	public void evaluateScaledScorecardAutoNA() throws Exception {
		evaluate("ScaledScorecard", AUTO_NA);
	}

	@Test
	public void evaluateBinningProcessIris() throws Exception {
		evaluate("BinningProcess", IRIS);
	}
}