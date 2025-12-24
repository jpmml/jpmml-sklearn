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

public class ThresholdClassifierUtil {

	private ThresholdClassifierUtil(){
	}

	static
	public Model encodeModel(Classifier estimator, Number threshold, Schema schema){
		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();

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
}