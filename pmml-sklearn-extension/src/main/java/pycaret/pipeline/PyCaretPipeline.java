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
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import pycaret.preprocess.TransformerWrapper;
import sklearn.Estimator;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;
import sklearn.pipeline.Pipeline;
import sklearn2pmml.pipeline.PMMLPipeline;

public class PyCaretPipeline extends Pipeline implements Encodable {

	public PyCaretPipeline(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public List<? extends TransformerWrapper> getTransformers(){
		List<? extends Transformer> transformers = super.getTransformers();

		return Lists.transform(transformers, TransformerWrapper.class::cast);
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
	public PMML encodePMML(){
		SkLearnEncoder encoder = new SkLearnEncoder();

		List<? extends TransformerWrapper> transformers = getTransformers();
		Estimator estimator = getFinalEstimator();

		TransformerWrapper transformer = transformers.get(0);

		String targetName = transformer.getTargetName();
		if(targetName != null){
			Label label = PMMLPipeline.initLabel(estimator, Collections.singletonList(targetName), encoder);

			encoder.setLabel(label);
		}

		Schema schema = encoder.createSchema();

		Model model = encodeModel(schema);

		encoder.setModel(model);

		return encoder.encodePMML(model);
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