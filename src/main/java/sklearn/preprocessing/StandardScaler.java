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
import sklearn.Transformer;

public class StandardScaler extends Transformer {

	public StandardScaler(String module, String name){
		super(module, name);
	}

	@Override
	public Expression encode(FieldName name){
		Number mean = Iterables.getOnlyElement(getMean());
		Number std = Iterables.getOnlyElement(getStd());

		// (name - mean) / std
		return PMMLUtil.createApply("/", PMMLUtil.createApply("-", new FieldRef(name), PMMLUtil.createConstant(mean)), PMMLUtil.createConstant(std));
	}

	public List<? extends Number> getMean(){
		return (List)ClassDictUtil.getArray(this, "mean_");
	}

	public List<? extends Number> getStd(){
		return (List)ClassDictUtil.getArray(this, "std_");
	}
}