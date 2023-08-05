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
package org.jpmml.sklearn.testing;

import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ResultField;
import org.jpmml.python.testing.PythonEncoderBatch;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.SkLearnUtil;
import sklearn.HasPMMLOptions;
import sklearn.Step;

abstract
public class SkLearnEncoderBatch extends PythonEncoderBatch {

	public SkLearnEncoderBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		super(algorithm, dataset, columnFilter, equivalence);
	}

	@Override
	abstract
	public SkLearnEncoderBatchTest getArchiveBatchTest();

	@Override
	public PMML getPMML() throws Exception {
		Map<String, ?> options = getOptions();

		Step step = loadPickle(Step.class);

		activate(step);

		try {
			Encodable encodable = EncodableUtil.toEncodable(step);

			if(options != null && !options.isEmpty()){
				HasPMMLOptions<?> hasPmmlOptions = (HasPMMLOptions<?>)encodable;

				hasPmmlOptions.setPMMLOptions(options);
			}

			PMML pmml = encodable.encodePMML();

			validatePMML(pmml);

			return pmml;
		} finally {
			deactivate(step);
		}
	}

	protected void activate(Object object) throws Exception {
	}

	protected void deactivate(Object object) throws Exception {
	}

	static {
		SkLearnUtil.initOnce();
	}
}