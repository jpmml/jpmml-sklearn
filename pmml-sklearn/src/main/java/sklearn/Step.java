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

import org.jpmml.python.PythonObject;
import sklearn2pmml.SkLearn2PMMLFields;

abstract
public class Step extends PythonObject implements HasNumberOfFeatures, HasType {

	public Step(String module, String name){
		super(module, name);
	}

	public void checkVersion(){
		checkSkLearnVersion();
	}

	public void checkSkLearnVersion(){
		String sklearnVersion = getSkLearnVersion();

		if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "1.2.2") > 0){
			String message = "This converter version does not know about Scikit-Learn version " + sklearnVersion + " artifacts. " +
				"Please upgrade the converter to the latest version, or downgrade Scikit-Learn to version 1.2.2";

			throw new IllegalArgumentException(message);
		}
	}

	public String getPMMLName(){
		return getOptionalString(SkLearn2PMMLFields.PMML_NAME);
	}

	public Step setPMMLName(String name){
		put(SkLearn2PMMLFields.PMML_NAME, name);

		return this;
	}

	public List<String> getSkLearnFeatureNamesIn(){

		if(containsKey(SkLearnFields.FEATURE_NAMES_IN)){
			return (List<String>)getArray(SkLearnFields.FEATURE_NAMES_IN, String.class);
		}

		return null;
	}

	public String getSkLearnVersion(){
		return getOptionalString(SkLearnFields.SKLEARN_VERSION);
	}
}