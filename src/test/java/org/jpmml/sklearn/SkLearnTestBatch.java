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
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import com.google.common.io.ByteStreams;
import h2o.estimators.BaseEstimator;
import org.dmg.pmml.Array;
import org.dmg.pmml.Constant;
import org.dmg.pmml.HasValue;
import org.dmg.pmml.HasValueSet;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.IntegrationTestBatch;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.python.CompressedInputStreamStorage;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.Storage;
import sklearn.Estimator;
import sklearn2pmml.pipeline.PMMLPipeline;

import static org.junit.Assert.assertFalse;

abstract
public class SkLearnTestBatch extends IntegrationTestBatch {

	private Map<String, Object> options = new LinkedHashMap<>();


	public SkLearnTestBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		super(name, dataset, predicate, equivalence);
	}

	@Override
	abstract
	public SkLearnTest getIntegrationTest();

	@Override
	public PMML getPMML() throws Exception {
		SkLearnEncoder encoder = new SkLearnEncoder();

		PMMLPipeline pipeline;

		try(Storage storage = openStorage("/pkl/" + getName() + getDataset() + ".pkl")){
			pipeline = (PMMLPipeline)PickleUtil.unpickle(storage);
		}

		Map<String, ?> options = getOptions();

		Estimator estimator = pipeline.getFinalEstimator();

		if(!options.isEmpty()){
			estimator.putOptions(options);
		}

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

	@Override
	protected void validatePMML(PMML pmml) throws Exception {
		super.validatePMML(pmml);

		Visitor visitor = new AbstractVisitor(){

			@Override
			public VisitorAction visit(PMMLObject object){

				if(object instanceof HasValue){
					checkValue((HasValue<?>)object);
				} // End if

				if(object instanceof HasValueSet){
					checkValueSet((HasValueSet<?>)object);
				}

				return super.visit(object);
			}

			@Override
			public VisitorAction visit(Constant constant){
				Object value = constant.getValue();

				assertFalse(isNaN(value));

				return super.visit(constant);
			}

			private void checkValue(HasValue<?> hasValue){
				Object value = hasValue.getValue();

				assertFalse(isNaN(value));
			}

			private void checkValueSet(HasValueSet<?> hasValueSet){
				Array array = hasValueSet.getArray();

				Object arrayValue = array.getValue();

				if(arrayValue instanceof Collection){
					Collection<?> values = (Collection<?>)arrayValue;

					for(Object value : values){
						assertFalse(isNaN(value));
					}
				}
			}

			private boolean isNaN(Object value){

				if(value instanceof Number){
					Number number = (Number)value;

					return Double.isNaN(number.doubleValue());
				}

				return false;
			}
		};
		visitor.applyTo(pmml);
	}

	public Map<String, ?> getOptions(){
		return this.options;
	}

	public void putOptions(Map<String, ?> options){
		this.options.putAll(options);
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