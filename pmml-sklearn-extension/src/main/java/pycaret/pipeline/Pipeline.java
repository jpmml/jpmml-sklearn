/*
 * Copyright (c) 2023 Villu Ruusmann
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
package pycaret.pipeline;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.Model;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasNumberOfFeatures;

public class Pipeline extends sklearn.pipeline.Pipeline {

	public Pipeline(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Feature> result = super.encodeFeatures(features, encoder);

		Label label = encoder.getLabel();

		if(label != null){
			result = new ArrayList<>(result);

			List<ScalarLabel> scalarLabels = ScalarLabelUtil.toScalarLabels(label);
			for(ScalarLabel scalarLabel : scalarLabels){
				Feature labelFeature = FeatureUtil.findLabelFeature(result, scalarLabel);

				if(labelFeature != null){
					result.remove(labelFeature);
				}
			}
		}

		return result;
	}

	@Override
	public Model encodeModel(Schema schema){
		return super.encodeModel(schema);
	}

	@Override
	public Label refreshLabel(Label label, SkLearnEncoder encoder){

		// XXX
		if(label instanceof ScalarLabel){
			ScalarLabel scalarLabel = (ScalarLabel)label;

			if(!scalarLabel.isAnonymous()){
				DataField dataField = (DataField)encoder.getField(scalarLabel.getName());

				return ScalarLabelUtil.createScalarLabel(dataField);
			}
		}

		return label;
	}
}