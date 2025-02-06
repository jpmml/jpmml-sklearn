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

import org.jpmml.sklearn.SkLearnException;
import org.junit.jupiter.api.Test;
import sklearn.Classifier;
import sklearn.CompositeClassifier;
import sklearn.CompositeRegressor;
import sklearn.Estimator;
import sklearn.PassThrough;
import sklearn.Regressor;
import sklearn.SkLearnSteps;
import sklearn.dummy.DummyClassifier;
import sklearn.dummy.DummyRegressor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class SkLearnPipelineTest {

	@Test
	public void emptyPipeline(){
		SkLearnPipeline pipeline = new SkLearnPipeline()
			.setSteps(Collections.emptyList());

		assertFalse(pipeline.hasTransformers());
		assertEquals(Collections.emptyList(), pipeline.getTransformers());

		try {
			pipeline.getHead();

			fail();
		} catch(SkLearnException se){
			// Ignored
		}

		assertFalse(pipeline.hasFinalEstimator());

		try {
			pipeline.getFinalEstimator();

			fail();
		} catch(SkLearnException se){
			// Ignored
		}
	}

	@Test
	public void nonePipeline(){
		SkLearnPipeline pipeline = new SkLearnPipeline()
			.setSteps(Collections.singletonList(new Object[]{"step", null}));

		checkIdentityTransform(pipeline);
	}

	@Test
	public void passthroughPipeline(){
		SkLearnPipeline pipeline = new SkLearnPipeline()
			.setSteps(Collections.singletonList(new Object[]{"step", SkLearnSteps.PASSTHROUGH}));

		checkIdentityTransform(pipeline);
	}

	@Test
	public void classifierPipeline(){
		SkLearnPipeline pipeline = forEstimator(new DummyClassifier());

		Classifier classifier = pipeline.toClassifier();

		assertTrue(classifier instanceof CompositeClassifier);
	}

	@Test
	public void regressorPipeline(){
		SkLearnPipeline pipeline = forEstimator(new DummyRegressor());

		Regressor regressor = pipeline.toRegressor();

		assertTrue(regressor instanceof CompositeRegressor);
	}

	static
	private void checkIdentityTransform(SkLearnPipeline pipeline){
		assertTrue(pipeline.hasTransformers());
		assertEquals(Collections.singletonList(PassThrough.INSTANCE), pipeline.getTransformers());

		assertNull(pipeline.getHead());

		assertFalse(pipeline.hasFinalEstimator());

		try {
			pipeline.getFinalEstimator();

			fail();
		} catch(SkLearnException se){
			// Ignored
		}
	}

	static
	private SkLearnPipeline forEstimator(Estimator estimator){
		List<Object[]> steps = Arrays.asList(
			new Object[]{"first", SkLearnSteps.PASSTHROUGH},
			new Object[]{"second", estimator}
		);

		SkLearnPipeline pipeline = new SkLearnPipeline()
			.setSteps(steps);

		assertTrue(pipeline.hasTransformers());
		assertEquals(Collections.singletonList(PassThrough.INSTANCE), pipeline.getTransformers());

		assertTrue(pipeline.hasFinalEstimator());
		assertEquals(estimator, pipeline.getFinalEstimator());

		return pipeline;
	}
}