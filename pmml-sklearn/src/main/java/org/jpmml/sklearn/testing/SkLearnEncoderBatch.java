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

import java.util.Collections;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import net.razorvine.pickle.Unpickler;
import org.dmg.pmml.PMML;
import org.jpmml.converter.PMMLEncodable;
import org.jpmml.evaluator.ResultField;
import org.jpmml.python.testing.PythonEncoderBatch;
import org.jpmml.sklearn.SkLearnUnpickler;
import org.jpmml.sklearn.SkLearnUtil;
import sklearn.Estimator;
import sklearn.pipeline.SkLearnPipeline;
import sklearn.tree.Tree;
import sklearn.tree.TreeUtil;

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
		if(options == null){
			options = Collections.emptyMap();
		}

		Object object = loadPickle(Object.class);

		Number recordCount = getRecordCount();

		if(recordCount != null){
			Tree tree = findFirstTree(object);

			if(tree != null){
				double[] weightedNodeSamples = tree.getWeightedNodeSamples();

				assertRecordCount(weightedNodeSamples[0]);
			}
		}

		activate(object);

		try {
			PMMLEncodable encodable = SkLearnUtil.toEncodable(object);

			PMML pmml = SkLearnUtil.encodePMML(encodable, options);

			validatePMML(pmml);

			return pmml;
		} finally {
			deactivate(object);
		}
	}

	@Override
	public Unpickler getUnpickler(){
		return new SkLearnUnpickler();
	}

	protected void activate(Object object) throws Exception {
	}

	protected void deactivate(Object object) throws Exception {
	}

	static
	private Tree findFirstTree(Object object){

		if(object instanceof Estimator){
			Estimator estimator = (Estimator)object;

			return TreeUtil.findFirstTree(estimator);
		} else

		if(object instanceof SkLearnPipeline){
			SkLearnPipeline pipeline = (SkLearnPipeline)object;

			return TreeUtil.findFirstTree(pipeline.getFinalEstimator());
		}

		return null;
	}

	static {
		SkLearnUtil.initOnce();
	}
}