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
package sklearn2pmml.postprocessing;

import java.util.List;

import org.dmg.pmml.Field;
import org.dmg.pmml.Model;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.IdentityTransformer;

public class FeatureExporter extends IdentityTransformer {

	public FeatureExporter(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<String> names = getNames();

		if(!encoder.hasModel()){
			throw new IllegalStateException("Model is not defined");
		}

		Model model = encoder.getModel();

		SchemaUtil.checkSize(names.size(), features);

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			String name = names.get(i);

			Field<?> field = feature.getField();

			OutputField outputField = new OutputField(name, field.requireOpType(), field.requireDataType())
				.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
				.setFinalResult(true)
				.setExpression(feature.ref());

			encoder.createDerivedField(model, outputField, true);
		}

		return features;
	}

	public List<String> getNames(){
		return getStringListLike("names");
	}
}