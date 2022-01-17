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
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
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
import org.jpmml.evaluator.visitors.DefaultModelEvaluatorBattery;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.python.testing.PythonEncoderBatch;
import sklearn.Estimator;
import sklearn2pmml.pipeline.PMMLPipeline;

import static org.junit.Assert.assertFalse;

abstract
public class SkLearnTestBatch extends PythonEncoderBatch {

	public SkLearnTestBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		super(algorithm, dataset, columnFilter, equivalence);
	}

	@Override
	abstract
	public SkLearnTest getArchiveBatchTest();

	@Override
	public PMML getPMML() throws Exception {
		String algorithm = getAlgorithm();

		SkLearnEncoder encoder = new SkLearnEncoder();

		PMMLPipeline pipeline = (PMMLPipeline)loadPickle();

		Map<String, ?> options = getOptions();

		Estimator estimator = pipeline.getFinalEstimator();

		if(!options.isEmpty()){
			estimator.putOptions(options);
		}

		File tmpFile = null;

		if(estimator instanceof BaseEstimator){
			BaseEstimator baseEstimator = (BaseEstimator)estimator;

			tmpFile = File.createTempFile(getAlgorithm() + getDataset(), ".mojo.zip");

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

		// XXX
		if(algorithm.equals(SkLearnAlgorithms.RIDGE_ENSEMBLE)){
			DefaultModelEvaluatorBattery visitorBattery = new DefaultModelEvaluatorBattery();
			visitorBattery.applyTo(pmml);
		} // End if

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
				Array array = hasValueSet.requireArray();

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
}