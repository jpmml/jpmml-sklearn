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

public class PMMLPipelineTest {

	@Test
	public void construct(){
		PMMLPipeline pipeline = new PMMLPipeline()
			.setSteps(Collections.<Object[]>emptyList());

		assertNull(pipeline.getRepr());
		assertNull(pipeline.getActiveFields());
		assertNull(pipeline.getTargetFields());

		Estimator estimator = new DummyClassifier(null, null);

		pipeline
			.setSteps(Collections.singletonList(new Object[]{"estimator", estimator}))
			.setRepr("PMMLPipeline([steps=(\"estimator\", DummyClassifier())])")
			.setTargetFields(Collections.singletonList("y"))
			.setActiveFields(Arrays.asList("x1", "x2", "x3"));

		assertEquals(Collections.emptyList(), pipeline.getTransformers());
		assertEquals(estimator, pipeline.getEstimator());

		assertNotNull(pipeline.getRepr());
		assertEquals(Arrays.asList("y"), pipeline.getTargetFields());
		assertEquals(Arrays.asList("x1", "x2", "x3"), pipeline.getActiveFields());
	}
}