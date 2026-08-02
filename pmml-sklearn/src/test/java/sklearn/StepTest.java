/*
 * Copyright (c) 2026 Villu Ruusmann
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
package sklearn;

import java.util.ArrayList;
import java.util.List;

import sklearn.pipeline.SkLearnPipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

abstract
public class StepTest {

	static
	protected SkLearnPipeline createPipeline(String name, Step step){
		SkLearnPipeline pipeline = new SkLearnPipeline()
			.setOnlyStep(name, step);

		return pipeline;
	}

	static
	protected void checkParents(List<Step> expectedParents, List<Step> parents){
		assertEquals(expectedParents.size(), parents.size());

		for(int i = 0; i < expectedParents.size(); i++){
			assertSame(expectedParents.get(i), parents.get(i));
		}
	}

	static
	protected List<Step> collectParents(Step step){
		List<Step> result = new ArrayList<>();

		Step parent = step.getParent();

		while(parent != null){
			result.add(parent);

			parent = parent.getParent();
		}

		return result;
	}
}
