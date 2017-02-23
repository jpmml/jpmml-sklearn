/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.linear_model.logistic;

import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Schema;
import sklearn.linear_model.BaseLinearClassifier;

public class LogisticRegression extends BaseLinearClassifier {

	public LogisticRegression(String module, String name){
		super(module, name);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		String multiClass = getMultiClass();

		if(!("ovr").equals(multiClass)){
			throw new IllegalArgumentException(multiClass);
		}

		return super.encodeModel(schema);
	}

	public String getMultiClass(){
		return (String)get("multi_class");
	}
}