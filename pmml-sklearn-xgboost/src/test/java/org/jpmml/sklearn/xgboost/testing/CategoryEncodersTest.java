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
package org.jpmml.sklearn.xgboost.testing;

import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class CategoryEncodersTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	public CategoryEncodersTest(){
		super(new FloatEquivalence(TOLERANCE_AUDIT));
	}

	@Test
	public void evaluateBase4EncoderAudit() throws Exception {
		evaluate("Base4Encoder", AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(TOLERANCE_AUDIT + 32));
	}

	@Test
	public void evaluateBase4EncoderAuditNA() throws Exception {
		evaluate("Base4Encoder", AUDIT_NA, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(TOLERANCE_AUDIT + 8));
	}

	@Test
	public void evaluateBinaryEncoderAuditNA() throws Exception {
		evaluate("BinaryEncoder", AUDIT_NA, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(TOLERANCE_AUDIT + 16));
	}

	@Test
	public void evaluateCatBoostEncoderAuditNA() throws Exception {
		evaluate("CatBoostEncoder", AUDIT_NA, excludeFields(AUDIT_PROBABILITY_FALSE));
	}

	@Test
	public void evaluateCountEncoderAuditNA() throws Exception {
		evaluate("CountEncoder", AUDIT_NA, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(TOLERANCE_AUDIT + 32));
	}

	@Test
	public void evaluateLeaveOneOutEncoderAuditNA() throws Exception {
		evaluate("LeaveOneOutEncoder", AUDIT_NA, excludeFields(AUDIT_PROBABILITY_FALSE));
	}

	@Test
	public void evaluateTargetEncoderAuditNA() throws Exception {
		evaluate("TargetEncoder", AUDIT_NA, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(TOLERANCE_AUDIT + 24));
	}

	@Test
	public void evaluateWOEEncoderAuditNA() throws Exception {
		evaluate("WOEEncoder", AUDIT_NA, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(TOLERANCE_AUDIT + 24));
	}

	private static final int TOLERANCE_AUDIT = 32 + 32;
}