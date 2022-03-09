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
import org.jpmml.model.visitors.VisitorBattery;
import org.jpmml.python.testing.PythonEncoderBatch;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn2pmml.pipeline.PMMLPipeline;

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
		SkLearnEncoder encoder = new SkLearnEncoder();

		PMMLPipeline pipeline = (PMMLPipeline)loadPickle();

		activate(pipeline);

		try {
			PMML pmml = pipeline.encodePMML(encoder);

			validatePMML(pmml);

			return pmml;
		} finally {
			deactivate(pipeline);
		}
	}

	protected void activate(PMMLPipeline pipeline) throws Exception {
		Map<String, ?> options = getOptions();

		Estimator estimator = pipeline.getFinalEstimator();

		if(!options.isEmpty()){
			estimator.putOptions(options);
		}
	}

	protected void deactivate(PMMLPipeline pipeline) throws Exception {
	}

	@Override
	public VisitorBattery getValidators(){
		VisitorBattery visitorBattery = super.getValidators();

		visitorBattery.add(ValueInspector.class);

		return visitorBattery;
	}
}