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
package sklearn;

import java.util.List;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ValueUtil;

public class ClassifierUtil {

	private ClassifierUtil(){
	}

	static
	public List<?> getClasses(Estimator estimator){
		HasClasses hasClasses = (HasClasses)estimator;

		return hasClasses.getClasses();
	}

	static
	public List<String> formatTargetCategories(List<?> objects){
		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				String targetCategory = ValueUtil.formatValue(object);

				if(targetCategory == null || CharMatcher.WHITESPACE.matchesAnyOf(targetCategory)){
					throw new IllegalArgumentException(targetCategory);
				}

				return targetCategory;
			}
		};

		return Lists.transform(objects, function);
	}

	static
	public void checkSize(int size, CategoricalLabel categoricalLabel){

		if(categoricalLabel.size() != size){
			throw new IllegalArgumentException("Expected " + size + " class(es), got " + categoricalLabel.size() + " class(es)");
		}
	}
}
