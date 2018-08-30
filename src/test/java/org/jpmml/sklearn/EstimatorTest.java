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
package org.jpmml.sklearn;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.io.ByteStreams;
import h2o.estimators.BaseEstimator;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Batch;
import org.jpmml.evaluator.IntegrationTest;
import org.jpmml.evaluator.IntegrationTestBatch;
import org.jpmml.evaluator.PMMLEquivalence;
import sklearn.Estimator;
import sklearn2pmml.pipeline.PMMLPipeline;

abstract
public class EstimatorTest extends IntegrationTest {

	public EstimatorTest(){
		super(new PMMLEquivalence(1e-13, 1e-13));
	}

	@Override
	protected Batch createBatch(String name, String dataset, Predicate<FieldName> predicate){
		Batch result = new IntegrationTestBatch(name, dataset, predicate){

			@Override
			public IntegrationTest getIntegrationTest(){
				return EstimatorTest.this;
			}

			private Storage openStorage(String path) throws IOException {
				InputStream is = open(path);

				try {
					return new CompressedInputStreamStorage(is);
				} catch(IOException ioe){
					is.close();

					throw ioe;
				}
			}

			@Override
			public PMML getPMML() throws IOException {
				PMMLPipeline pipeline;

				try(Storage storage = openStorage("/pkl/" + getName() + getDataset() + ".pkl")){
					pipeline = (PMMLPipeline)PickleUtil.unpickle(storage);
				}

				Estimator estimator = pipeline.getEstimator();

				File tmpFile = null;

				if(estimator instanceof BaseEstimator){
					BaseEstimator baseEstimator = (BaseEstimator)estimator;

					tmpFile = File.createTempFile(getName() + getDataset(), ".mojo.zip");

					String mojoPath = baseEstimator.getMojoPath();

					try(InputStream is = open("/" + mojoPath)){

						try(OutputStream os = new FileOutputStream(tmpFile)){
							ByteStreams.copy(is, os);
						}
					}

					baseEstimator.setMojoPath(tmpFile.getAbsolutePath());
				}

				PMML pmml = pipeline.encodePMML();

				ensureValidity(pmml);

				if(tmpFile != null){
					tmpFile.delete();
				}

				return pmml;
			}

			@Override
			public List<Map<FieldName, String>> getInput() throws IOException {
				String dataset = super.getDataset();

				if(dataset.endsWith("Cat")){
					dataset = dataset.substring(0, dataset.length() - "Cat".length());
				} else

				if(dataset.endsWith("Dict")){
					dataset = dataset.substring(0, dataset.length() - "Dict".length());
				}

				return loadRecords("/csv/" + dataset + ".csv");
			}
		};

		return result;
	}
}