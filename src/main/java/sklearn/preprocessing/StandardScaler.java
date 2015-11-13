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

public class StandardScaler extends MultiTransformer {

	public StandardScaler(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getMeanShape();

		if(shape.length != 1){
			throw new IllegalArgumentException();
		}

		return shape[0];
	}

	@Override
	public Expression encode(int index, FieldName name){
		Expression expression = new FieldRef(name);

		if(withMean()){
			Number mean = Iterables.get(getMean(), index);

			if(Double.compare(mean.doubleValue(), 0d) != 0){
				expression = PMMLUtil.createApply("-", expression, PMMLUtil.createConstant(mean));
			}
		} // End if

		if(withStd()){
			Number std = Iterables.get(getStd(), index);

			if(Double.compare(std.doubleValue(), 1d) != 0){
				expression = PMMLUtil.createApply("/", expression, PMMLUtil.createConstant(std));
			}
		}

		// "($name - mean) / std"
		return expression;
	}

	public Boolean withMean(){
		return (Boolean)get("with_mean");
	}

	public Boolean withStd(){
		return (Boolean)get("with_std");
	}

	public List<? extends Number> getMean(){
		return (List)ClassDictUtil.getArray(this, "mean_");
	}

	public List<? extends Number> getStd(){
		try {
			// SkLearn 0.16
			return (List)ClassDictUtil.getArray(this, "std_");
		} catch(IllegalArgumentException iae){
			// SkLearn 0.17
			return (List)ClassDictUtil.getArray(this, "scale_");
		}
	}

	private int[] getMeanShape(){
		return ClassDictUtil.getShape(this, "mean_");
	}
}