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
package org.jpmml.sklearn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;
import sklearn.HasNumberOfFeatures;
import sklearn.HasNumberOfOutputs;
import sklearn.SkLearnFields;
import sklearn.Step;
import sklearn2pmml.pipeline.PMMLPipelineUtil;

public class EncodableUtil {

	private EncodableUtil(){
	}

	static
	public Encodable toEncodable(Object object){

		if(object instanceof Encodable){
			Encodable encodable = (Encodable)object;

			return encodable;
		}

		return PMMLPipelineUtil.toPMMLPipeline(object);
	}

	static
	public List<String> getOrGenerateFeatureNames(Step step){
		List<String> names = step.getFeatureNamesIn();

		if(names == null){
			return generateFeatureNames(step);
		}

		return names;
	}

	static
	public List<String> generateFeatureNames(Step step){
		int numberOfFeatures = step.getNumberOfFeatures();

		if(numberOfFeatures == HasNumberOfFeatures.UNKNOWN){
			throw new IllegalArgumentException("Attribute \'" + ClassDictUtil.formatMember(step, SkLearnFields.N_FEATURES_IN) + "\' is not set");
		}

		return generateNames("x", numberOfFeatures, true);
	}

	static
	public List<String> generateOutputNames(Estimator estimator){
		int numberOfOutputs = estimator.getNumberOfOutputs();

		if(numberOfOutputs == HasNumberOfOutputs.UNKNOWN){
			throw new IllegalArgumentException("Attribute \'" + ClassDictUtil.formatMember(estimator, SkLearnFields.N_OUTPUTS) + "\' is not set");
		}

		return generateNames("y", numberOfOutputs, false);
	}

	static
	private List<String> generateNames(String name, int count, boolean indexed){

		if(count <= 0){
			throw new IllegalArgumentException();
		} else

		if(count == 1){
			return Collections.singletonList(name + (indexed ? "1" : ""));
		} else

		{
			List<String> result = new ArrayList<>(count);

			for(int i = 0; i < count; i++){
				result.add(name + String.valueOf(i + 1));
			}

			return result;
		}
	}
}