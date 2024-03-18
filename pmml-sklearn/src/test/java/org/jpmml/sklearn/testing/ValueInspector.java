/*
 * Copyright (c) 2022 Villu Ruusmann
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
package org.jpmml.sklearn.testing;

import java.util.Collection;

import org.dmg.pmml.Array;
import org.dmg.pmml.Constant;
import org.dmg.pmml.HasValue;
import org.dmg.pmml.HasValueSet;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.visitors.AbstractVisitor;

public class ValueInspector extends AbstractVisitor {

	@Override
	public VisitorAction visit(PMMLObject object){

		if(object instanceof HasValue){
			checkValue((HasValue<?>)object);
		} // End if

		if(object instanceof HasValueSet){
			checkValueSet((HasValueSet<?>)object);
		}

		return super.visit(object);
	}

	@Override
	public VisitorAction visit(Constant constant){
		Object value = constant.getValue();

		if(isNonFinite(value)){
			throw new AssertionError();
		}

		return super.visit(constant);
	}

	private void checkValue(HasValue<?> hasValue){
		Object value = hasValue.getValue();

		if(isNonFinite(value)){
			throw new AssertionError();
		}
	}

	private void checkValueSet(HasValueSet<?> hasValueSet){
		Array array = hasValueSet.requireArray();

		Object arrayValue = array.getValue();

		if(arrayValue instanceof Collection){
			Collection<?> values = (Collection<?>)arrayValue;

			for(Object value : values){

				if(isNonFinite(value)){
					throw new AssertionError();
				}
			}
		}
	}

	private boolean isNonFinite(Object value){

		if(value instanceof Number){
			Number number = (Number)value;

			return !Double.isFinite(number.doubleValue());
		}

		return false;
	}
}