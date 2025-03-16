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

import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.FieldNames;
import org.jpmml.converter.FieldUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.AttributeException;
import org.jpmml.python.ClassDictUtil;
import sklearn.Classifier;
import sklearn.EstimatorUtil;
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

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		Model model = estimator.encodeModel(schema);

		Output output = EstimatorUtil.getFinalOutput(model);
		if(output == null){
			throw new IllegalArgumentException();
		}

		// XXX
		String name = FieldNameUtil.create(FieldNames.PROBABILITY, categoricalLabel.getValue(1));

		Expression expression = ExpressionUtil.createApply(PMMLFunctions.IF,
			ExpressionUtil.createApply(PMMLFunctions.LESSTHAN, new FieldRef(name), ExpressionUtil.createConstant(threshold)),
			ExpressionUtil.createConstant(categoricalLabel.getDataType(), categoricalLabel.getValue(0)), ExpressionUtil.createConstant(categoricalLabel.getDataType(), categoricalLabel.getValue(1))
		);

		OutputField thresholdedOutputField = new OutputField(FieldNameUtil.create("thresholded", categoricalLabel.getName()), categoricalLabel.getOpType(), categoricalLabel.getDataType())
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(expression);

		FieldUtil.addValues(thresholdedOutputField, categoricalLabel.getValues());

		output.addOutputFields(thresholdedOutputField);

		return model;
	}

	@Override
	public Classifier getEstimator(){
		return get("estimator_", Classifier.class);
	}

	public String getResponseMethod(){
		return getEnum("response_method", this::getString, Arrays.asList(FixedThresholdClassifier.RESPONSEMETHOD_PREDICT_PROBA));
	}

	public Number getThreshold(){
		Object threshold = getObject("threshold");

		if(Objects.equals(FixedThresholdClassifier.THRESHOLD_AUTO, threshold)){
			throw new AttributeException("Attribute \'" + ClassDictUtil.formatMember(this, "threshold") + "\' must be set to a numeric value");
		}

		return getNumber("threshold");
	}

	private static final String RESPONSEMETHOD_PREDICT_PROBA = "predict_proba";

	private static final String THRESHOLD_AUTO = "auto";
}