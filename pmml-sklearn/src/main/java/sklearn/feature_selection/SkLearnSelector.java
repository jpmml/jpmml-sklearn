/*
 * Copyright (c) 2018 Villu Ruusmann
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
package sklearn.feature_selection;

import java.util.List;

import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnException;
import sklearn.HasFeatureNamesIn;
import sklearn.Selector;
import sklearn2pmml.SelectorProxy;

public class SkLearnSelector extends Selector implements HasFeatureNamesIn {

	public SkLearnSelector(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		throw new SkLearnException(formatMessage());
	}

	@Override
	public List<Boolean> getSupportMask(){
		throw new SkLearnException(formatMessage());
	}

	private String formatMessage(){
		return "The selector object (" + ClassDictUtil.formatClass(this) + ") does not have persistent state. " +
			"Please use the " + (SelectorProxy.class).getName() + " wrapper class to give the selector object a persistent state (eg. " + SelectorProxy.formatProxyExample(this) + ")";
	}
}