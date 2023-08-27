/*
 * Copyright (c) 2023 Villu Ruusmann
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
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;

public class HyperoptTest extends SkLearnEncoderBatchTest implements Datasets {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public HyperoptTest getArchiveBatchTest(){
				return HyperoptTest.this;
			}

			@Override
			public List<? extends Map<String, ?>> getInput() throws IOException {
				List<? extends Map<String, ?>> records = super.getInput();

				String dataset = getDataset();
				if((AUDIT).equals(dataset) || (AUTO).equals(dataset)){
					records = anonymizeColumns(records);
				}

				return records;
			}
		};

		return result;
	}

	@Test
	public void evaluateHyperoptAudit() throws Exception {
		evaluate("Hyperopt", AUDIT);
	}

	@Test
	public void evaluateHyperoptAuto() throws Exception {
		evaluate("Hyperopt", AUTO);
	}

	@Test
	public void evaluateHyperoptIris() throws Exception {
		evaluate("Hyperopt", IRIS);
	}

	static
	private List<? extends Map<String, ?>> anonymizeColumns(List<? extends Map<String, ?>> records){
		List<Map<String, ?>> result = new ArrayList<>();

		for(Map<String, ?> record : records){
			result.add(anonymizeColumns(record));
		}

		return result;
	}

	static
	private Map<String, ?> anonymizeColumns(Map<String, ?> record){
		Map<String, Object> result = new LinkedHashMap<>();

		Collection<? extends Map.Entry<String, ?>> entries = record.entrySet();
		for(Iterator<? extends Map.Entry<String, ?>> it = entries.iterator(); it.hasNext(); ){
			Map.Entry<String, ?> entry = it.next();

			if(!it.hasNext()){
				break;
			}

			result.put("x" + (result.size() + 1), entry.getValue());
		}

		return result;
	}
}