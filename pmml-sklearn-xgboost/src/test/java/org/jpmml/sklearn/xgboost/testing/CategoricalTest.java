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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.junit.jupiter.api.Test;

public class CategoricalTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public CategoricalTest getArchiveBatchTest(){
				return CategoricalTest.this;
			}

			@Override
			public String getInputCsvPath(){
				String path = super.getInputCsvPath();

				path = path.replace("Cat", "");

				return path;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				Map<String, Object> options = new LinkedHashMap<>();
				options.put(HasXGBoostOptions.OPTION_COMPACT, false);

				return OptionsUtil.generateOptionsMatrix(options);
			}
		};

		return result;
	}

	@Test
	public void evaluateXGBAuditCat() throws Exception {
		evaluate("XGB", AUDIT + "Cat", excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(48 + 8)); // XXX
	}

	@Test
	public void evaluateXGBAuditCatNA() throws Exception {
		evaluate("XGB", AUDIT + "CatNA", excludeFields(AUDIT_PROBABILITY_TRUE), new FloatEquivalence(56 + 8)); // XXX
	}

	@Test
	public void evaluateXGBAutoCat() throws Exception {
		evaluate("XGB", AUTO + "Cat", new FloatEquivalence(8 + 4));
	}

	@Test
	public void evaluateMultiXGBAutoCat() throws Exception {
		evaluate("MultiXGB", AUTO + "Cat", new FloatEquivalence(12));
	}

	@Test
	public void evaluateXGBAutoCatNA() throws Exception {
		evaluate("XGB", AUTO + "CatNA", excludeFields(AUTO_MPG), new FloatEquivalence(8)); // XXX
	}

	@Test
	public void evaluateXGBRFAuditCat() throws Exception {
		evaluate("XGBRF", AUDIT + "Cat", excludeFields(AUDIT_PROBABILITY_FALSE), new FloatEquivalence(24));
	}

	@Test
	public void evaluateXGBRFAuditCatNA() throws Exception {
		evaluate("XGBRF", AUDIT + "CatNA", excludeFields(AUDIT_PROBABILITY_TRUE), new FloatEquivalence(16 + 4));
	}

	@Test
	public void evaluateXGBRFAutoCat() throws Exception {
		evaluate("XGBRF", AUTO + "Cat", new FloatEquivalence(8 + 2));
	}

	@Test
	public void evaluateMultiXGBRFAutoCat() throws Exception {
		evaluate("MultiXGBRF", AUTO + "Cat", new FloatEquivalence(32 + 8));
	}

	@Test
	public void evaluateXGBRFAutoCatNA() throws Exception {
		evaluate("XGBRF", AUTO + "CatNA", new FloatEquivalence(16)); // XXX
	}
}