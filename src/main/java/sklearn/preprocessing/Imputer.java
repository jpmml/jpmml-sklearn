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

public class Imputer extends Transformer {

	public Imputer(String module, String name){
		super(module, name);
	}

	@Override
	public Expression encode(FieldName name){
		Number statistics = Iterables.getOnlyElement(getStatistics());

		// (name == null) ? statistics : name
		return PMMLUtil.createApply("if", PMMLUtil.createApply("isMissing", new FieldRef(name)), PMMLUtil.createConstant(statistics), new FieldRef(name));
	}

	public List<? extends Number> getStatistics(){
		return (List)ClassDictUtil.getArray(this, "statistics_");
	}
}