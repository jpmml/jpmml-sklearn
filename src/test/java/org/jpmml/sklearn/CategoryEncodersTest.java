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
package org.jpmml.sklearn;

import org.dmg.pmml.FieldName;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.junit.Test;

public class CategoryEncodersTest extends SkLearnTest {

	@Test
	public void evaluateBase2EncoderAudit() throws Exception {
		evaluate("Base2Encoder", "Audit");
	}

	@Test
	public void evaluateBase3EncoderAudit() throws Exception {
		evaluate("Base3Encoder", "Audit");
	}

	@Test
	public void evaluateBase4EncoderAudit() throws Exception {
		evaluate("Base4Encoder", "Audit", excludeFields(CategoryEncodersTest.falseProbabilityField), new FloatEquivalence(8));
	}

	@Test
	public void evaluateBinaryEncoderAudit() throws Exception {
		evaluate("BinaryEncoder", "Audit");
	}

	@Test
	public void evaluateOrdinalEncoderAudit() throws Exception {
		evaluate("OrdinalEncoder", "Audit");
	}

	private static final FieldName falseProbabilityField = FieldNameUtil.create("probability", "0");
	private static final FieldName trueProbabilityField = FieldNameUtil.create("probability", "1");
}