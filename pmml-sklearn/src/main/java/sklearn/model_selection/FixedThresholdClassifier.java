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
import java.util.Objects;

import org.dmg.pmml.Model;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.Attribute;
import org.jpmml.python.InvalidAttributeException;
import sklearn.Classifier;
import sklearn.HasEstimator;
import sklearn.SkLearnClassifier;

public class FixedThresholdClassifier extends SkLearnClassifier implements HasEstimator<Classifier> {

	public FixedThresholdClassifier(String module, String name){
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
		Number threshold = getThreshold();

		return ThresholdClassifierUtil.encodeModel(estimator, threshold, schema);
	}

	@Override
	public Classifier getEstimator(){
		return getClassifier("estimator_");
	}

	public String getResponseMethod(){
		return getEnum("response_method", this::getString, Arrays.asList(FixedThresholdClassifier.RESPONSEMETHOD_PREDICT_PROBA));
	}

	public Number getThreshold(){
		Object threshold = getObject("threshold");

		if(Objects.equals(FixedThresholdClassifier.THRESHOLD_AUTO, threshold)){
			throw new InvalidAttributeException("Attribute " + ExceptionUtil.formatName("threshold") + " must be set to a numeric value", new Attribute(this, "threshold"));
		}

		return getNumber("threshold");
	}

	private static final String RESPONSEMETHOD_PREDICT_PROBA = "predict_proba";

	private static final String THRESHOLD_AUTO = "auto";
}