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
package causalml;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.EstimatorUtil;
import sklearn.Regressor;

public class CausalMLUtil {

	private CausalMLUtil(){
	}

	static
	public Schema toClassifierSchema(Classifier classifier, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		CategoricalLabel categoricalLabel = ((CategoricalLabel)classifier.encodeLabel(Collections.singletonList(null), encoder))
			.expectCardinality(2);

		return schema.toRelabeledSchema(categoricalLabel);
	}

	static
	public Schema toRegressorSchema(Regressor regressor, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		ContinuousLabel continuousLabel = (ContinuousLabel)regressor.encodeLabel(Collections.singletonList(null), encoder);

		return schema.toRelabeledSchema(continuousLabel);
	}

	static
	public OutputField getProbabilityField(Model model){
		Output output = EstimatorUtil.getFinalOutput(model);

		if(output == null || !output.hasOutputFields()){
			throw new IllegalArgumentException();
		}

		List<OutputField> outputFields = output.getOutputFields();
		if(outputFields.size() != 2){
			throw new IllegalArgumentException();
		}

		return Iterables.getLast(outputFields);
	}
}