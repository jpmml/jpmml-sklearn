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

import java.util.Collection;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.jpmml.python.AttributeException;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonObject;
import org.jpmml.sklearn.SkLearnException;
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

		if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "1.4.1") > 0){
			String message = "This converter version does not know about Scikit-Learn version " + sklearnVersion + " artifacts. " +
				"Please upgrade the converter to the latest version, or downgrade Scikit-Learn to version " + "1.4.1";

			throw new SkLearnException(message);
		}
	}

	public List<String> getFeatureNamesIn(){

		if(hasattr(SkLearnFields.FEATURE_NAMES_IN)){
			return getStringArray(SkLearnFields.FEATURE_NAMES_IN);
		}

		return null;
	}

	public String getPMMLName(){
		return getOptionalString(SkLearn2PMMLFields.PMML_NAME);
	}

	public Step setPMMLName(String pmmlName){
		setattr(SkLearn2PMMLFields.PMML_NAME, pmmlName);

		return this;
	}

	public String getSkLearnVersion(){
		return getOptionalString(SkLearnFields.SKLEARN_VERSION);
	}

	public <E> List<E> getEnumList(String name, Function<String, List<E>> function, Collection<E> enumValues){
		List<E> values = function.apply(name);

		Function<E, E> enumFunction = new Function<E, E>(){

			@Override
			public E apply(E value){

				if(!enumValues.contains(value)){
					// XXX
					throw new AttributeException("List attribute \'" + ClassDictUtil.formatMember(Step.this, name) + "\' contains an unsupported value " + String.valueOf(value) + ". Supported values are " + formatValues(enumValues));
				}

				return value;
			}
		};

		return Lists.transform(values, enumFunction);
	}
}