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
package org.jpmml.sklearn.lightgbm.testing;

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.RealNumberEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;

public class SkLearnLightGBMTest extends SkLearnEncoderBatchTest implements Datasets {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public SkLearnLightGBMTest getArchiveBatchTest(){
				return SkLearnLightGBMTest.this;
			}

			@Override
			public String getInputCsvPath(){
				String path = super.getInputCsvPath();

				path = path.replace("Cat", "");

				return path;
			}
		};

		return result;
	}

	@Test
	public void evaluateLGBMAudit() throws Exception {
		evaluate("LGBM", AUDIT, new RealNumberEquivalence(2));
	}

	@Test
	public void evaluateLGBMAuditCat() throws Exception {
		evaluate("LGBM", AUDIT + "Cat", new RealNumberEquivalence(2));
	}

	@Test
	public void evaluateLGBMLRAuditCat() throws Exception {
		evaluate("LGBMLR", AUDIT + "Cat");
	}

	@Test
	public void evaluateLGBMAuto() throws Exception {
		evaluate("LGBM", AUTO, new RealNumberEquivalence(1));
	}

	@Test
	public void evaluateLGBMIris() throws Exception {
		evaluate("LGBM", IRIS, new RealNumberEquivalence(1));
	}

	@Test
	public void evaluateLGBMIrisCat() throws Exception {
		evaluate("LGBM", IRIS + "Cat");
	}

	@Test
	public void evaluateLGBMIsotonicIris() throws Exception {
		evaluate("LGBM" + "Isotonic", IRIS);
	}

	@Test
	public void evaluateLGBMSigmoidVersicolor() throws Exception {
		evaluate("LGBM" + "Sigmoid", VERSICOLOR);
	}
}