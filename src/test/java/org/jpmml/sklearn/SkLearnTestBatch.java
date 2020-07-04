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
package org.jpmml.sklearn;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import com.google.common.io.ByteStreams;
import h2o.estimators.BaseEstimator;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.IntegrationTestBatch;
import org.jpmml.python.CompressedInputStreamStorage;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.Storage;
import sklearn.Estimator;
import sklearn2pmml.pipeline.PMMLPipeline;

abstract
public class SkLearnTestBatch extends IntegrationTestBatch {

	public SkLearnTestBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		super(name, dataset, predicate, equivalence);
	}

	@Override
	abstract
	public SkLearnTest getIntegrationTest();

	public Map<String, ?> getOptions(){
		return new LinkedHashMap<>();
	}

	@Override
	public PMML getPMML() throws Exception {
		SkLearnEncoder encoder = new SkLearnEncoder();

		PMMLPipeline pipeline;

		try(Storage storage = openStorage("/pkl/" + getName() + getDataset() + ".pkl")){
			pipeline = (PMMLPipeline)PickleUtil.unpickle(storage);
		}

		Map<String, ?> options = getOptions();

		Estimator estimator = pipeline.getFinalEstimator();

		Map<String, ?> pmmlOptions = estimator.getPMMLOptions();

		// Programmatic test batch options (empty by default) override pickle file options
		if(pmmlOptions != null){
			pmmlOptions.putAll((Map)options);

			options = pmmlOptions;
		}

		estimator.setPMMLOptions(options);

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

		PMML pmml = pipeline.encodePMML(encoder);

		validatePMML(pmml);

		if(tmpFile != null){
			tmpFile.delete();
		}

		return pmml;
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
}