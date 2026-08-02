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
package sklearn.model_selection;

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
import sklearn.Classifier;
import sklearn.SkLearnMethods;
import sklearn.Step;
import sklearn.StepTest;
import sklearn.pipeline.SkLearnPipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ThresholdClassifierTest extends StepTest {

	@Test
	public void encodeThresholdClassifier(){
		List<Step> parents = new ArrayList<>();
		List<String> stages = new ArrayList<>();

		Classifier classifier = new Classifier(null, null){

			@Override
			public int getNumberOfFeatures(){
				return 1;
			}

			@Override
			public List<?> getClasses(){
				return Arrays.asList(0, 1);
			}

			@Override
			public RegressionModel encodeModel(Schema schema){
				List<? extends Feature> features = schema.getFeatures();

				stages.add("encodeModel");

				checkParents(2, this);

				parents.addAll(collectParents(this));

				return RegressionModelUtil.createBinaryLogisticClassification(features, Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.LOGIT, true, schema);
			}

			@Override
			public Schema configureSchema(Schema schema){
				stages.add("configureSchema");

				checkParents(2, this);

				return schema;
			}

			@Override
			public Model configureModel(Model model){
				stages.add("configureModel");

				checkParents(2, this);

				return model;
			}
		};

		ThresholdClassifier thresholdClassifier = new ThresholdClassifier(null, null){

			@Override
			public int getNumberOfFeatures(){
				return 1;
			}

			@Override
			public Classifier getEstimator(){
				return classifier;
			}

			@Override
			public String getResponseMethod(){
				return SkLearnMethods.PREDICT_PROBA;
			}

			@Override
			public Number getThreshold(){
				return 0.5d;
			}
		};

		SkLearnPipeline pipeline = createPipeline("thresholdClassifier", thresholdClassifier);

		pipeline.encodePMML();

		assertEquals(Arrays.asList("configureSchema", "encodeModel", "configureModel"), stages);

		checkParents(Arrays.asList(thresholdClassifier, pipeline), parents);
	}
}
