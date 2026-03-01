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
import java.util.Objects;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.jupiter.api.Test;

public class CausalMLTest extends SkLearnEncoderBatchTest {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public CausalMLTest getArchiveBatchTest(){
				return CausalMLTest.this;
			}

			@Override
			public Table getInput() throws IOException {
				Table table = super.getInput();

				String algorithm = getAlgorithm();
				String dataset = getDataset();

				if(Objects.equals(dataset, "Email")){

					if(Objects.equals(algorithm, "DecisionTreeSRegressor") || Objects.equals(algorithm, "GradientBoostingSRegressor") || Objects.equals(algorithm, "RandomForestSRegressor")){
						return subTable(table, 4000);
					}
				}

				return table;
			}
		};

		return result;
	}

	@Test
	public void evaluateDecisionTreeSRegressorEmail() throws Exception {
		evaluate("DecisionTreeSRegressor", "Email");
	}

	@Test
	public void evaluateGradientBoostingSRegressorEmail() throws Exception {
		evaluate("GradientBoostingSRegressor", "Email");
	}

	@Test
	public void evaluateRandomForestSRegressorEmail() throws Exception {
		evaluate("RandomForestSRegressor", "Email");
	}

	static
	private Table subTable(Table table, int rows){
		Table result = new Table(table.getColumns(), rows);

		Table.Row readerRow = table.createReaderRow(0, rows);
		Table.Row writerRow = result.createWriterRow(0);

		while(readerRow.canAdvance()){
			writerRow.putAll(readerRow);

			readerRow.advance();
			writerRow.advance();
		}

		return result;
	}
}