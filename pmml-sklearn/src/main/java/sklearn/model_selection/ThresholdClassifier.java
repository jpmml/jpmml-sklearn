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

import java.util.Arrays;
import java.util.List;

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
import sklearn.Classifier;
import sklearn.EstimatorUtil;
import sklearn.HasEstimator;
import sklearn.SkLearnClassifier;
import sklearn.SkLearnMethods;
import sklearn.Step;

abstract
public class ThresholdClassifier extends SkLearnClassifier implements HasEstimator<Classifier> {

	public ThresholdClassifier(String module, String name){
		super(module, name);
	}

	abstract
	public Number getThreshold();

	@Override
	public List<?> getClasses(){
		Classifier estimator = getEstimator();

		return estimator.getClasses();
	}

	@Override
	public Model encode(Step parent, Schema schema){
		Classifier classifier = getEstimator();

		Step prevParent = classifier.getParent();

		try {
			classifier.setParent(this);

			return super.encode(parent, schema);
		} finally {
			classifier.setParent(prevParent);
		}
	}

	@Override
	public Model encodeModel(Schema schema){
		@SuppressWarnings("unused")
		String responseMethod = getResponseMethod();
		Number threshold = getThreshold();

		Classifier classifier = getEstimator();

		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel()
			.expectCardinality(2);

		Model model = classifier.encodeModel(schema);

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
	public Schema configureSchema(Schema schema){
		Classifier classifier = getEstimator();

		return classifier.configureSchema(schema);
	}

	@Override
	public Model configureModel(Model model){
		Classifier classifier = getEstimator();

		return classifier.configureModel(model);
	}

	@Override
	public Classifier getEstimator(){
		return getClassifier("estimator_");
	}

	public String getResponseMethod(){
		return getEnum("response_method", this::getString, Arrays.asList(SkLearnMethods.PREDICT_PROBA));
	}
}
