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

import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ArchiveBatch;
import org.jpmml.evaluator.ArchiveBatchTest;
import sklearn.Estimator;
import sklearn_pandas.DataFrameMapper;

abstract
public class EstimatorTest extends ArchiveBatchTest {

	@Override
	protected ArchiveBatch createBatch(String name, String dataset){
		ArchiveBatch result = new ArchiveBatch(name, dataset){

			@Override
			public InputStream open(String path){
				Class<? extends EstimatorTest> clazz = EstimatorTest.this.getClass();

				return clazz.getResourceAsStream(path);
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
				PMML pmml;

				Schema schema;

				Storage estimatorStorage = openStorage("/pkl/" + getName() + getDataset() + ".pkl");

				try {
					Estimator estimator = (Estimator)PickleUtil.unpickle(estimatorStorage);

					schema = estimator.createSchema();

					pmml = estimator.encodePMML(schema);
				} finally {
					estimatorStorage.close();
				}

				Storage mapperStorage = openStorage("/pkl/" + getDataset() + ".pkl");

				try {
					DataFrameMapper mapper = (DataFrameMapper)PickleUtil.unpickle(mapperStorage);

					mapper.updatePMML(schema, pmml);
				} finally {
					mapperStorage.close();
				}

				return pmml;
			}
		};

		return result;
	}
}