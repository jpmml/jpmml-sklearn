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
package sklearn;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.PMML;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.SkLearnEncoder;

public class TransformerUtil {

	private TransformerUtil(){
	}

	static
	public PMML encodePMML(Transformer transformer){
		SkLearnEncoder encoder = new SkLearnEncoder();

		List<Feature> features = new ArrayList<>();

		if(!(transformer instanceof Initializer)){
			List<String> activeFields = StepUtil.getOrGenerateFeatureNames(transformer);

			features = encoder.initFeatures(transformer, activeFields);
		}

		features = transformer.encode(Step.PARENT_ROOT, features, encoder);

		encoder.setFeatures(features);

		return encoder.encodePMML(null);
	}
}