/*
 * Copyright (c) 2020 Villu Ruusmann
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

import java.util.List;

import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class StepUtil {

	private StepUtil(){
	}

	static
	public void checkNumberOfFeatures(Step step, List<? extends Feature> features){
		int numberOfFeatures = step.getNumberOfFeatures();

		if((numberOfFeatures != HasNumberOfFeatures.UNKNOWN) && (numberOfFeatures != features.size())){
			throw new IllegalArgumentException("Expected " + numberOfFeatures + " feature(s) (" + ClassDictUtil.formatClass(step)  + "), got " + features.size() + " feature(s)");
		}
	}

	static
	public int getNumberOfFeatures(List<? extends Step> steps){

		for(Step step : steps){
			return step.getNumberOfFeatures();
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	static
	public List<String> getOrGenerateFeatureNames(Step step){
		List<String> result = step.getFeatureNamesIn();

		if(result == null){
			result = generateFeatureNames(step);
		}

		return result;
	}

	static
	public List<String> generateFeatureNames(Step step){
		int numberOfFeatures = step.getNumberOfFeatures();

		if(numberOfFeatures == HasNumberOfFeatures.UNKNOWN){
			throw new IllegalArgumentException("Attribute \'" + ClassDictUtil.formatMember(step, SkLearnFields.N_FEATURES_IN) + "\' is not set");
		}

		return SkLearnEncoder.generateNames("x", numberOfFeatures, true);
	}
}