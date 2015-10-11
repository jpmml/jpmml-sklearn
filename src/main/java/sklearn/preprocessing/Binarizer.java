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

import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.jpmml.converter.PMMLUtil;
import sklearn.Transformer;

public class Binarizer extends Transformer {

	public Binarizer(String module, String name){
		super(module, name);
	}

	@Override
	public Expression encode(FieldName name){
		Number threshold = getThreshold();

		// "($name <= threshold) ? 0 : 1"
		return PMMLUtil.createApply("if", PMMLUtil.createApply("lessOrEqual", new FieldRef(name), PMMLUtil.createConstant(threshold)), PMMLUtil.createConstant(0), PMMLUtil.createConstant(1));
	}

	public Number getThreshold(){
		return (Number)get("threshold");
	}
}