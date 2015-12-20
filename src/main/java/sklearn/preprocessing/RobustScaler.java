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
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.MultiTransformer;

public class RobustScaler extends MultiTransformer {

	public RobustScaler(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getCenterOrScaleShape();

		return shape[0];
	}

	@Override
	public Expression encode(int index, FieldName name){
		Expression expression = new FieldRef(name);

		if(getWithCentering()){
			Number center = Iterables.get(getCenter(), index);

			if(!ValueUtil.isZero(center)){
				expression = PMMLUtil.createApply("-", expression, PMMLUtil.createConstant(center));
			}
		} // End if

		if(getWithScaling()){
			Number scale = Iterables.get(getScale(), index);

			if(!ValueUtil.isOne(scale)){
				expression = PMMLUtil.createApply("/", expression, PMMLUtil.createConstant(scale));
			}
		}

		// "($name - center) / scale"
		return expression;
	}

	public Boolean getWithCentering(){
		return (Boolean)get("with_centering");
	}

	public Boolean getWithScaling(){
		return (Boolean)get("with_scaling");
	}

	public List<? extends Number> getCenter(){
		return (List)ClassDictUtil.getArray(this, "center_");
	}

	public List<? extends Number> getScale(){
		return (List)ClassDictUtil.getArray(this, "scale_");
	}

	private int[] getCenterOrScaleShape(){

		if(getWithCentering()){
			return ClassDictUtil.getShape(this, "center_", 1);
		} else

		if(getWithScaling()){
			return ClassDictUtil.getShape(this, "scale_", 1);
		}

		return new int[]{1};
	}
}