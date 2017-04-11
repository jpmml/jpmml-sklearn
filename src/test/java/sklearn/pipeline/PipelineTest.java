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
package sklearn.pipeline;

import java.util.Collections;
import java.util.List;

import org.junit.Test;
import sklearn.Estimator;
import sklearn.Transformer;
import sklearn.dummy.DummyClassifier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

public class PipelineTest {

	@Test
	public void construct(){
		Pipeline pipeline = new Pipeline();
		pipeline.put("steps", Collections.emptyList());

		assertEquals(Collections.emptyList(), pipeline.getTransformers());
		assertNull(pipeline.getEstimator());

		Estimator estimator = new DummyClassifier(null, null);

		pipeline.put("steps", Collections.singletonList(new Object[]{"estimator", estimator}));

		try {
			List<Transformer> transformers = pipeline.getTransformers();

			// Consume elements
			for(Transformer transformer : transformers){
				assertNotNull(transformer);
			}

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}

		assertNull(pipeline.getEstimator());
	}
}