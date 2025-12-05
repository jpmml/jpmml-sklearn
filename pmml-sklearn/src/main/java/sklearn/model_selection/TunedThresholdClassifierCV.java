/*
 * Copyright (c) 2025 Villu Ruusmann
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

import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import sklearn.Classifier;
import sklearn.HasEstimator;
import sklearn.SkLearnClassifier;

public class TunedThresholdClassifierCV extends SkLearnClassifier implements HasEstimator<Classifier> {

	public TunedThresholdClassifierCV(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		Classifier estimator = getEstimator();

		return estimator.getClasses();
	}

	@Override
	public Model encodeModel(Schema schema){
		Classifier estimator = getEstimator();
		@SuppressWarnings("unused")
		String responseMethod = getResponseMethod();
		Number bestThreshold = getBestThreshold();

		return ThresholdClassifierUtil.encodeModel(estimator, bestThreshold, schema);
	}

	@Override
	public Classifier getEstimator(){
		return getClassifier("estimator_");
	}

	public Number getBestThreshold(){
		return getNumber("best_threshold_");
	}

	public String getResponseMethod(){
		return getEnum("response_method", this::getString, Arrays.asList(TunedThresholdClassifierCV.RESPONSEMETHOD_PREDICT_PROBA));
	}

	private static final String RESPONSEMETHOD_PREDICT_PROBA = "predict_proba";
}