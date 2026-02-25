/*
 * Copyright (c) 2026 Villu Ruusmann
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

import java.io.IOException;
import java.util.Collections;
import java.util.Objects;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import ngboost.NGBoostNames;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class NGBoostTest extends SkLearnEncoderBatchTest implements Datasets {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public NGBoostTest getArchiveBatchTest(){
				return NGBoostTest.this;
			}

			@Override
			public Table getInput() throws IOException {
				Table table = super.getInput();

				String dataset = getDataset();

				if(Objects.equals(dataset, AUTO)){
					int numberOfRows = table.getNumberOfRows();

					table.setValues(NGBoostNames.INPUT_CI, Collections.nCopies(numberOfRows, 0.95));
				}

				return table;
			}
		};

		return result;
	}

	@Test
	public void evaluateNGBoostAudit() throws Exception {
		evaluate("NGBoost", AUDIT);
	}

	@Test
	public void evaluateNGBoostAuto() throws Exception {
		evaluate("NGBoost", AUTO);
	}

	@Test
	public void evaluateNGBoostLogAuto() throws Exception {
		evaluate("NGBoostLog", AUTO);
	}

	@Test
	public void evaluateNGBoostIris() throws Exception {
		evaluate("NGBoost", IRIS);
	}

	@Test
	public void evaluateNGBoostLung() throws Exception {
		evaluate("NGBoost", "Lung");
	}

	@Test
	public void evaluateNGBoostVisit() throws Exception {
		evaluate("NGBoost", VISIT);
	}
}