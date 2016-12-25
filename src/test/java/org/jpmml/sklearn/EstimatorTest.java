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

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.Application;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.evaluator.Batch;
import org.jpmml.evaluator.IntegrationTest;
import org.jpmml.evaluator.IntegrationTestBatch;
import org.jpmml.evaluator.visitors.InvalidFeatureInspector;
import org.jpmml.evaluator.visitors.UnsupportedFeatureInspector;
import sklearn2pmml.PMMLPipeline;

abstract
public class EstimatorTest extends IntegrationTest {

	@Override
	protected Batch createBatch(String name, String dataset){
		Batch result = new IntegrationTestBatch(name, dataset){

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

				PMML pmml = pipeline.encodePMML();

				ensureValidity(pmml);

				return pmml;
			}
		};

		return result;
	}

	static
	private void ensureValidity(PMML pmml){
		List<Visitor> visitors = Arrays.<Visitor>asList(
			new UnsupportedFeatureInspector(),
			new InvalidFeatureInspector(){

				@Override
				public VisitorAction visit(Application application){
					return VisitorAction.SKIP;
				}

				@Override
				public VisitorAction visit(MiningSchema miningSchema){
					return VisitorAction.SKIP;
				}
			}
		);

		for(Visitor visitor : visitors){
			visitor.applyTo(pmml);
		}
	}

	static {

		try {
			PickleUtil.init();
		} catch(Exception e){
			throw new AssertionError(e);
		}
	}
}