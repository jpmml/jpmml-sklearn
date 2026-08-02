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

import org.dmg.pmml.Model;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.junit.jupiter.api.Test;
import sklearn.compose.TransformedTargetRegressor;
import sklearn.pipeline.SkLearnPipeline;
import sklearn.preprocessing.FunctionTransformer;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class EstimatorTest extends StepTest {

	@Test
	public void encodeTransformedTargetRegressor(){
		List<Step> parents = new ArrayList<>();

		Regressor regressor = new Regressor(null, null){

			{
				putOption("overridden", "regressor");
			}

			@Override
			public RegressionModel encodeModel(Schema schema){
				List<? extends Feature> features = schema.getFeatures();

				assertEquals(1, features.size());

				checkOptions();

				parents.addAll(collectParents(this));

				return RegressionModelUtil.createRegression(features, Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.NONE, schema);
			}

			@Override
			public Schema configureSchema(Schema schema){
				checkOptions();

				return schema;
			}

			@Override
			public Model configureModel(Model model){
				checkOptions();

				return model;
			}

			private void checkOptions(){
				assertEquals("transformedTargetRegressor", getOption("inherited", null));
				assertEquals("regressor", getOption("overridden", null));
				assertEquals("fallback", getOption("undeclared", "fallback"));
			}
		};

		TransformedTargetRegressor transformedTargetRegressor = new TransformedTargetRegressor(null, null){

			{
				putOption("inherited", "transformedTargetRegressor");
				putOption("overridden", "transformedTargetRegressor");
			}

			@Override
			public int getNumberOfFeatures(){
				return 1;
			}

			@Override
			public Regressor getRegressor(){
				return regressor;
			}

			@Override
			public FunctionTransformer getTransformer(){
				return new FunctionTransformer(null, null);
			}
		};

		SkLearnPipeline pipeline = createPipeline("transformedTargetRegressor", transformedTargetRegressor);

		pipeline.encodePMML();

		checkParents(Arrays.asList(transformedTargetRegressor, pipeline), parents);
	}
}
