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
package sklearn2pmml.pipeline;

import java.util.Arrays;
import java.util.Collections;

import org.junit.jupiter.api.Test;
import sklearn.Estimator;
import sklearn.dummy.DummyClassifier;
import sklearn.pipeline.SkLearnPipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

public class PMMLPipelineTest {

	@Test
	public void construct(){
		PMMLPipeline pipeline = new PMMLPipeline()
			.setSteps(Collections.emptyList());

		assertNull(pipeline.getRepr());
		assertNull(pipeline.getActiveFields());
		assertNull(pipeline.getTargetFields());

		Estimator estimator = new DummyClassifier();

		pipeline
			.setOnlyStep("estimator", estimator)
			.setRepr("PMMLPipeline([steps=(\"estimator\", DummyClassifier())])")
			.setTargetFields(Collections.singletonList("y"))
			.setActiveFields(Arrays.asList("x1", "x2", "x3"));

		assertEquals(Collections.emptyList(), pipeline.getTransformers());
		assertEquals(estimator, pipeline.getFinalEstimator());

		assertNotNull(pipeline.getRepr());
		assertEquals(Arrays.asList("y"), pipeline.getTargetFields());
		assertEquals(Arrays.asList("x1", "x2", "x3"), pipeline.getActiveFields());
	}


	@Test
	public void configure(){
		Estimator estimator = new DummyClassifier();

		PMMLPipeline pipeline = new PMMLPipeline()
			.setOnlyStep("estimator", estimator);

		assertNull(estimator.getPMMLOptions());

		pipeline.configure(Collections.singletonMap("flag", true));

		assertEquals(Collections.singletonMap("flag", true), estimator.getPMMLOptions());
	}

	@Test
	public void configureNested(){
		Estimator estimator = new DummyClassifier();

		SkLearnPipeline classifierPipeline = new SkLearnPipeline()
			.setOnlyStep("estimator", estimator);

		PMMLPipeline pipeline = new PMMLPipeline()
			.setOnlyStep("pipeline", classifierPipeline);

		assertNull(estimator.getPMMLOptions());

		pipeline.configure(Collections.singletonMap("flag", true));

		assertEquals(Collections.singletonMap("flag", true), estimator.getPMMLOptions());
	}
}