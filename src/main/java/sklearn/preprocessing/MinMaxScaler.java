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
package sklearn.preprocessing;

import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.MultiTransformer;
import sklearn.ValueUtil;

public class MinMaxScaler extends MultiTransformer {

	public MinMaxScaler(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getMinShape();

		return shape[0];
	}

	@Override
	public Expression encode(int index, FieldName name){
		Number min = Iterables.get(getMin(), index);
		Number scale = Iterables.get(getScale(), index);

		Expression expression = new FieldRef(name);

		if(!ValueUtil.isOne(scale)){
			expression = PMMLUtil.createApply("*", expression, PMMLUtil.createConstant(scale));
		} // End if

		if(!ValueUtil.isZero(min)){
			expression = PMMLUtil.createApply("+", expression, PMMLUtil.createConstant(min));
		}

		// "($name * scale) + min"
		return expression;
	}

	public List<? extends Number> getMin(){
		return (List)ClassDictUtil.getArray(this, "min_");
	}

	public List<? extends Number> getScale(){
		return (List)ClassDictUtil.getArray(this, "scale_");
	}

	private int[] getMinShape(){
		return ClassDictUtil.getShape(this, "min_", 1);
	}
}