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

import org.jpmml.converter.FieldNamePrefixes;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.sklearn.FieldNames;
import org.junit.Test;
import sklearn.Estimator;

public class SkLearn2PMMLTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Test
	public void evaluateCHAIDAudit() throws Exception {
		evaluate("CHAID", AUDIT, excludeFields(AUDIT_ADJUSTED, AUDIT_PROBABILITY_FALSE, AUDIT_PROBABILITY_TRUE));
	}

	@Test
	public void evaluateGBDTLRAudit() throws Exception {
		evaluate("GBDTLR", AUDIT);
	}

	@Test
	public void evaluateMultiEstimatorChainAudit() throws Exception {
		evaluate("MultiEstimatorChain", AUDIT, excludeFields(FieldNameUtil.create(FieldNamePrefixes.PROBABILITY, "Male"), FieldNameUtil.create(FieldNamePrefixes.PROBABILITY, "Female"), FieldNames.NODE_ID, AUDIT_PROBABILITY_FALSE, AUDIT_PROBABILITY_TRUE));
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
	public void evaluateExpressionAuto() throws Exception {
		evaluate("Expression", AUTO);
	}

	@Test
	public void evaluateGBDTLMAuto() throws Exception {
		evaluate("GBDTLM", AUTO);
	}

	@Test
	public void evaluateMultiEstimatorChainAuto() throws Exception {
		evaluate("MultiEstimatorChain", AUTO, excludeFields(FieldNameUtil.create(Estimator.FIELD_PREDICT, "acceleration"), FieldNames.NODE_ID));
	}

	@Test
	public void evaluateCHAIDAutoNA() throws Exception {
		evaluate("CHAID", AUTO_NA, excludeFields(AUTO_MPG));
	}

	@Test
	public void evaluateGBDTLMHousing() throws Exception {
		evaluate("GBDTLM", HOUSING);
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

	@Test
	public void evaluateRuleSetIris() throws Exception {
		evaluate("RuleSet", IRIS);
	}

	@Test
	public void evaluateSelectFirstIris() throws Exception {
		evaluate("SelectFirst", IRIS, excludeFields(IRIS_PROBABILITY_SETOSA, IRIS_PROBABILITY_VERSICOLOR, IRIS_PROBABILITY_VIRGINICA));
	}

	@Test
	public void evaluateGBDTLRVersicolor() throws Exception {
		evaluate("GBDTLR", VERSICOLOR);
	}
}