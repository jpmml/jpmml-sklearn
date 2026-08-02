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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.junit.jupiter.api.Test;
import sklearn.pipeline.SkLearnPipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

public class CompositeTest extends StepTest {

	@Test
	public void encodeTransformer(){
		List<Step> parents = new ArrayList<>();

		Transformer transformer = new Transformer(null, null){

			@Override
			public int getNumberOfFeatures(){
				return 1;
			}

			@Override
			public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
				assertEquals(1, features.size());

				parents.addAll(collectParents(this));

				return features;
			}
		};

		SkLearnPipeline transformerPipeline = createPipeline("transformer", transformer);

		SkLearnPipeline pipeline = createPipeline("pipeline", transformerPipeline);

		Transformer compositeTransformer = pipeline.toTransformer();

		assertInstanceOf(CompositeTransformer.class, compositeTransformer);

		pipeline.encodePMML();

		checkParents(Arrays.asList(transformerPipeline, pipeline), parents);
	}

	@Test
	public void encodeRegressor(){
		List<Step> parents = new ArrayList<>();

		Regressor regressor = new Regressor(null, null){

			@Override
			public int getNumberOfFeatures(){
				return 1;
			}

			@Override
			public RegressionModel encodeModel(Schema schema){
				List<? extends Feature> features = schema.getFeatures();

				assertEquals(1, features.size());

				parents.addAll(collectParents(this));

				return RegressionModelUtil.createRegression(features, Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.NONE, schema);
			}
		};

		SkLearnPipeline regressorPipeline = createPipeline("regressor", regressor);

		SkLearnPipeline pipeline = createPipeline("pipeline", regressorPipeline);

		Regressor compositeRegressor = pipeline.toRegressor();

		assertInstanceOf(CompositeRegressor.class, compositeRegressor);

		pipeline.encodePMML();

		checkParents(Arrays.asList(regressorPipeline, pipeline), parents);
	}
}
