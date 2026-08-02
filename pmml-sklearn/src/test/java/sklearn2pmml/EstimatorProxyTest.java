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
package sklearn2pmml;

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
import sklearn.Estimator;
import sklearn.Regressor;
import sklearn.Step;
import sklearn.StepTest;
import sklearn.pipeline.SkLearnPipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class EstimatorProxyTest extends StepTest {

	@Test
	public void encodeEstimatorProxy(){
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

				checkParents(2, this);

				parents.addAll(collectParents(this));

				return RegressionModelUtil.createRegression(features, Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.NONE, schema);
			}

			@Override
			public Schema configureSchema(Schema schema){
				checkParents(2, this);

				return schema;
			}

			@Override
			public Model configureModel(Model model){
				checkParents(2, this);

				return model;
			}

		};

		EstimatorProxy estimatorProxy = new EstimatorProxy(){

			@Override
			public Estimator getEstimator(){
				return regressor;
			}
		};

		SkLearnPipeline pipeline = createPipeline("estimatorProxy", estimatorProxy);

		pipeline.encodePMML();

		checkParents(Arrays.asList(estimatorProxy, pipeline), parents);
	}

	@Test
	public void encodeCompositeEstimatorProxy(){
		List<Step> parents = new ArrayList<>();

		Regressor regressor = new Regressor(null, null){

			@Override
			public int getNumberOfFeatures(){
				return 1;
			}

			@Override
			public RegressionModel encodeModel(Schema schema){
				List<? extends Feature> features = schema.getFeatures();

				parents.addAll(collectParents(this));

				return RegressionModelUtil.createRegression(features, Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.NONE, schema);
			}
		};

		SkLearnPipeline regressorPipeline = new SkLearnPipeline()
			.setOnlyStep("estimator", regressor);

		EstimatorProxy estimatorProxy = new EstimatorProxy();
		estimatorProxy.setattr("estimator", regressorPipeline);

		SkLearnPipeline pipeline = createPipeline("estimatorProxy", estimatorProxy);

		pipeline.encodePMML();

		checkParents(Arrays.asList(regressorPipeline, estimatorProxy, pipeline), parents);
	}

	static
	private void checkParents(int numberOfParents, Step step){
		List<Step> parents = collectParents(step);

		assertEquals(numberOfParents, parents.size());
	}
}
