/*
 * Copyright (c) 2015 Villu Ruusmann
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

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.ModelUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.jpmml.evaluator.testing.Conflict;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.python.testing.PythonEncoderBatchTest;

abstract
public class SkLearnEncoderBatchTest extends PythonEncoderBatchTest {

	private TableMutation inputMutation = null;

	private TableMutation outputMutation = null;

	private Table expectedOutput = null;


	public SkLearnEncoderBatchTest(){
		this(new PMMLEquivalence(1e-13, 1e-13));
	}

	public SkLearnEncoderBatchTest(Equivalence<Object> equivalence){
		super(equivalence);
	}

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public SkLearnEncoderBatchTest getArchiveBatchTest(){
				return SkLearnEncoderBatchTest.this;
			}
		};

		return result;
	}

	public void evaluate(String algorithm, String dataset, TableMutation inputMutation, TableMutation outputMutation) throws Exception {
		evaluate(algorithm, dataset, inputMutation, outputMutation, (resultField -> true), getEquivalence());
	}

	public void evaluate(String algorithm, String dataset, TableMutation inputMutation, TableMutation outputMutation, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence) throws Exception {
		this.inputMutation = inputMutation;
		this.outputMutation = outputMutation;

		try {
			evaluate(algorithm, dataset, columnFilter, equivalence);
		} finally {
			this.inputMutation = null;
			this.outputMutation = null;

			this.expectedOutput = null;
		}
	}

	@Override
	protected void checkConflicts(List<Conflict> conflicts){
		Table expectedOutput = this.expectedOutput;

		if(expectedOutput == null || !expectedOutput.hasExceptions()){
			super.checkConflicts(conflicts);

			return;
		}

		List<Conflict> unexpectedConflicts = new ArrayList<>();

		Set<Integer> matchedRows = new LinkedHashSet<>();

		for(Conflict conflict : conflicts){
			Integer row = (Integer)conflict.getId();

			Exception expectedException = expectedOutput.getException(row);
			if(expectedException != null){
				Exception exception = conflict.getException();

				if(exception != null && ExceptionAssignment.matches(expectedException, exception)){
					matchedRows.add(row);

					continue;
				}
			}

			unexpectedConflicts.add(conflict);
		}

		super.checkConflicts(unexpectedConflicts);

		for(int row = 0; row < expectedOutput.getNumberOfRows(); row++){
			Exception expectedException = expectedOutput.getException(row);

			if(expectedException != null && !matchedRows.contains(row)){
				throw new AssertionError("Expected " + expectedException + " at row " + row + ", got no exception");
			}
		}
	}

	public TableMutation getInputMutation(){
		return this.inputMutation;
	}

	public TableMutation getOutputMutation(){
		return this.outputMutation;
	}

	void setExpectedOutput(Table expectedOutput){
		this.expectedOutput = expectedOutput;
	}

	static
	protected String[] createNeighborFields(int count){
		List<OutputField> neighborFields = ModelUtil.createNeighborFields(count);

		return neighborFields.stream()
			.map(OutputField::requireName)
			.toArray(String[]::new);
	}
}