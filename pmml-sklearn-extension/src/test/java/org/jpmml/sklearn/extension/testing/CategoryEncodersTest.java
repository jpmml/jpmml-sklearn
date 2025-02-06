/*
 * Copyright (c) 2020 Villu Ruusmann
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
import org.junit.jupiter.api.Test;

public class CategoryEncodersTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Test
	public void evaluateBalancedDecisionTreeEnsembleAudit() throws Exception {
		evaluate("BalancedDecisionTreeEnsemble", AUDIT);
	}

	@Test
	public void evaluateBalancedRandomForestAudit() throws Exception {
		evaluate("BalancedRandomForest", AUDIT);
	}

	@Test
	public void evaluateBase2EncoderAudit() throws Exception {
		evaluate("Base2Encoder", AUDIT);
	}

	@Test
	public void evaluateBase2EncoderAuditNA() throws Exception {
		evaluate("Base2Encoder", AUDIT_NA);
	}

	@Test
	public void evaluateBase3EncoderAudit() throws Exception {
		evaluate("Base3Encoder", AUDIT);
	}

	@Test
	public void evaluateBase3EncoderAuditNA() throws Exception {
		evaluate("Base3Encoder", AUDIT_NA);
	}

	@Test
	public void evaluateBase4EncoderAudit() throws Exception {
		evaluate("Base4Encoder", AUDIT);
	}

	@Test
	public void evaluateBase4EncoderAuditNA() throws Exception {
		evaluate("Base4Encoder", AUDIT_NA);
	}

	@Test
	public void evaluateBinaryEncoderAudit() throws Exception {
		evaluate("BinaryEncoder", AUDIT);
	}

	@Test
	public void evaluateBinaryEncoderAuditNA() throws Exception {
		evaluate("BinaryEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateCatBoostEncoderAudit() throws Exception {
		evaluate("CatBoostEncoder", AUDIT);
	}

	@Test
	public void evaluateCatBoostEncoderAuditNA() throws Exception {
		evaluate("CatBoostEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateCountEncoderAudit() throws Exception {
		evaluate("CountEncoder", AUDIT);
	}

 	@Test
	public void evaluateCountEncoderAuditNA() throws Exception {
		evaluate("CountEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateLeaveOneOutEncoderAudit() throws Exception {
		evaluate("LeaveOneOutEncoder", AUDIT);
	}

	@Test
	public void evaluateLeaveOneOutEncoderAuditNA() throws Exception {
		evaluate("LeaveOneOutEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateOneHotEncoderAudit() throws Exception {
		evaluate("OneHotEncoder", AUDIT);
	}

	@Test
	public void evaluateOneHotEncoderAuditNA() throws Exception {
		evaluate("OneHotEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateOrdinalEncoderAudit() throws Exception {
		evaluate("OrdinalEncoder", AUDIT);
	}

	@Test
	public void evaluateOrdinalEncoderAuditNA() throws Exception {
		evaluate("OrdinalEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateTargetEncoderAudit() throws Exception {
		evaluate("TargetEncoder", AUDIT);
	}

	@Test
	public void evaluateTargetEncoderAuditNA() throws Exception {
		evaluate("TargetEncoder", AUDIT_NA);
	}

	@Test
	public void evaluateWOEEncoderAudit() throws Exception {
		evaluate("WOEEncoder", AUDIT);
	}

	@Test
	public void evaluateWOEEncoderAuditNA() throws Exception {
		evaluate("WOEEncoder", AUDIT_NA);
	}
}