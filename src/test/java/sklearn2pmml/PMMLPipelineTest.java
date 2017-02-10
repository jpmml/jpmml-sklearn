/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn2pmml;

import java.util.Arrays;
import java.util.Collections;

import org.junit.Test;
import sklearn.Estimator;
import sklearn.dummy.DummyClassifier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

public class PMMLPipelineTest {

	@Test
	public void construct(){
		PMMLPipeline pipeline = new PMMLPipeline();

		try {
			pipeline.getMapper();

			fail();
		} catch(IllegalArgumentException iae){
			// Ingored
		}

		try {
			pipeline.getTransformers();

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}

		try {
			pipeline.getEstimator();

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}

		assertNull(pipeline.getSteps());
		assertNull(pipeline.getRepr());
		assertNull(pipeline.getActiveFields());
		assertNull(pipeline.getTargetField());

		Estimator estimator = new DummyClassifier(null, null);

		pipeline
			.setSteps(Collections.singletonList(new Object[]{"estimator", estimator}))
			.setRepr("PMMLPipeline([steps=(\"estimator\", DummyClassifier())])")
			.setTargetField("y")
			.setActiveFields(Arrays.asList("x1", "x2", "x3"));

		assertEquals(null, pipeline.getMapper());
		assertEquals(Collections.emptyList(), pipeline.getTransformers());
		assertEquals(estimator, pipeline.getEstimator());

		assertEquals(estimator.getMiningFunction(), pipeline.getMiningFunction());
		assertEquals(estimator.isSupervised(), pipeline.isSupervised());
		assertEquals(estimator.getNumberOfFeatures(), pipeline.getNumberOfFeatures());
		assertEquals(estimator.getOpType(), pipeline.getOpType());
		assertEquals(estimator.getDataType(), pipeline.getDataType());

		assertNotNull(pipeline.getRepr());
		assertEquals("y", pipeline.getTargetField());
		assertEquals(Arrays.asList("x1", "x2", "x3"), pipeline.getActiveFields());
	}
}