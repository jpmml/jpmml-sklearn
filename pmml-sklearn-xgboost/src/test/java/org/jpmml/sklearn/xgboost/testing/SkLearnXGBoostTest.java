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

import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;
import sklearn.Estimator;

public class SkLearnXGBoostTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Test
	public void evaluateXGBAudit() throws Exception {
		evaluate("XGB", AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(8));
	}

	@Test
	public void evaluateXGBAuditNA() throws Exception {
		String[] transformFields = {AUDIT_PROBABILITY_FALSE, FieldNameUtil.create(Estimator.FIELD_PREDICT, AUDIT_ADJUSTED), FieldNameUtil.create("eval", AUDIT_ADJUSTED)};

		evaluate("XGB", AUDIT_NA, excludeFields(transformFields), new FloatEquivalence(8));
	}

	@Test
	public void evaluateXGBAuto() throws Exception {
		evaluate("XGB", AUTO, new FloatEquivalence(4));
	}

	@Test
	public void evaluateXGBIris() throws Exception {
		evaluate("XGB", IRIS, excludeFields(IRIS_PROBABILITY_SETOSA), new FloatEquivalence(16));
	}

	@Test
	public void evaluateXGBLRAudit() throws Exception {
		evaluate("XGBLR", AUDIT);
	}

	@Test
	public void evaluateXGBRFAudit() throws Exception {
		evaluate("XGBRF", AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(4));
	}

	@Test
	public void evaluateXGBRFAuto() throws Exception {
		evaluate("XGBRF", AUTO, new FloatEquivalence(2));
	}

	@Test
	public void evaluateXGBRFLMAuto() throws Exception {
		evaluate("XGBRFLM", AUTO);
	}

	@Test
	public void evaluateXGBRFLRAudit() throws Exception {
		evaluate("XGBRFLR", AUDIT);
	}

	@Test
	public void evaluateXGBRFLMHousing() throws Exception {
		evaluate("XGBRFLM", HOUSING);
	}

	@Test
	public void evaluateXGBRFLRVersicolor() throws Exception {
		evaluate("XGBRFLR", VERSICOLOR);
	}
}