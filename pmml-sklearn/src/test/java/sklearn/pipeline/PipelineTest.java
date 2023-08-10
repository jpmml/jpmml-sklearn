/*
 * Copyright (c) 2023 Villu Ruusmann
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
package sklearn.pipeline;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Test;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.PassThrough;
import sklearn.CompositeClassifier;
import sklearn.CompositeRegressor;
import sklearn.Regressor;
import sklearn.SkLearnSteps;
import sklearn.dummy.DummyClassifier;
import sklearn.dummy.DummyRegressor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class PipelineTest {

	@Test
	public void emptyPipeline(){
		Pipeline pipeline = new Pipeline()
			.setSteps(Collections.emptyList());

		assertFalse(pipeline.hasTransformers());
		assertEquals(Collections.emptyList(), pipeline.getTransformers());

		assertNull(pipeline.getHead());

		assertFalse(pipeline.hasFinalEstimator());

		try {
			pipeline.getFinalEstimator();

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}
	}

	@Test
	public void passthroughPipeline(){
		Pipeline pipeline = new Pipeline()
			.setSteps(Collections.singletonList(new Object[]{"estimator", SkLearnSteps.PASSTHROUGH}));

		assertFalse(pipeline.hasTransformers());
		assertEquals(Collections.emptyList(), pipeline.getTransformers());

		assertNull(pipeline.getHead());

		assertTrue(pipeline.hasFinalEstimator());
		assertEquals(null, pipeline.getFinalEstimator());
	}

	@Test
	public void classifierPipeline(){
		Pipeline pipeline = forEstimator(new DummyClassifier());

		Classifier classifier = pipeline.toClassifier();

		assertTrue(classifier instanceof CompositeClassifier);
	}

	@Test
	public void regressorPipeline(){
		Pipeline pipeline = forEstimator(new DummyRegressor());

		Regressor regressor = pipeline.toRegressor();

		assertTrue(regressor instanceof CompositeRegressor);
	}

	static
	private Pipeline forEstimator(Estimator estimator){
		List<Object[]> steps = Arrays.asList(
			new Object[]{"first", SkLearnSteps.PASSTHROUGH},
			new Object[]{"second", estimator}
		);

		Pipeline pipeline = new Pipeline()
			.setSteps(steps);

		assertTrue(pipeline.hasTransformers());
		assertEquals(Collections.singletonList(PassThrough.INSTANCE), pipeline.getTransformers());

		assertTrue(pipeline.hasFinalEstimator());
		assertEquals(estimator, pipeline.getFinalEstimator());

		return pipeline;
	}
}