/*
 * Copyright (c) 2018 Villu Ruusmann
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
package org.jpmml.sklearn.h2o.testing;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import com.google.common.io.ByteStreams;
import h2o.estimators.H2OEstimator;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.sklearn.testing.SkLearnEncoderBatch;
import org.jpmml.sklearn.testing.SkLearnEncoderBatchTest;
import org.junit.Test;
import sklearn.Composite;

public class SkLearnH2OTest extends SkLearnEncoderBatchTest implements Datasets, Fields {

	@Override
	public SkLearnEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SkLearnEncoderBatch result = new SkLearnEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public SkLearnH2OTest getArchiveBatchTest(){
				return SkLearnH2OTest.this;
			}

			@Override
			protected void activate(Object object) throws Exception {
				super.activate(object);

				H2OEstimator h2oEstimator = getEstimator(object);

				File tmpFile = File.createTempFile(getAlgorithm() + getDataset(), ".mojo.zip");

				try(InputStream is = open("/" + h2oEstimator.getMojoPath())){

					try(OutputStream os = new FileOutputStream(tmpFile)){
						ByteStreams.copy(is, os);
					}
				}

				h2oEstimator.setMojoPath(tmpFile.getAbsolutePath());
			}

			@Override
			protected void deactivate(Object object) throws Exception {
				super.deactivate(object);

				H2OEstimator h2oEstimator = getEstimator(object);

				File tmpFile = new File(h2oEstimator.getMojoPath());

				if(tmpFile.isFile()){
					tmpFile.delete();
				}
			}

			private H2OEstimator getEstimator(Object object){

				if(object instanceof Composite){
					Composite composite = (Composite)object;

					object = composite.getFinalEstimator();
				}

				return (H2OEstimator)object;
			}
		};

		return result;
	}

	@Test
	public void evaluateGradientBoostingAudit() throws Exception {
		String[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2OGradientBoosting", AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateGradientBoostingAuto() throws Exception {
		evaluate("H2OGradientBoosting", AUTO);
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		String[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2OLogisticRegression", AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("H2OLinearRegression", AUTO);
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		String[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2ORandomForest", AUDIT, excludeFields(targetFields));
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate("H2ORandomForest", AUTO);
	}

	@Test
	public void evaluateXGBoostAudit() throws Exception {
		String[] targetFields = createTargetFields(AUDIT_ADJUSTED);

		evaluate("H2OXGBoost", AUDIT, excludeFields(targetFields), new FloatEquivalence(16));
	}

	@Test
	public void evaluateXGBoostAuto() throws Exception {
		evaluate("H2OXGBoost", AUTO, new FloatEquivalence(8));
	}

	static
	private String[] createTargetFields(String name){
		return new String[]{name, FieldNameUtil.create("h2o", name)};
	}
}