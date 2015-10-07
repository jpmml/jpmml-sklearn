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

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Value;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;

abstract
public class Classifier extends Estimator {

	public Classifier(String module, String name){
		super(module, name);
	}

	@Override
	public DataField encodeTargetField(){
		DataField dataField = new DataField(FieldName.create("y"), OpType.CATEGORICAL, DataType.STRING);

		List<Value> values = dataField.getValues();

		List<?> classes = getClasses();

		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				return PMMLUtil.formatValue(object);
			}
		};

		values.addAll(PMMLUtil.createValues(Lists.transform((List<?>)classes, function)));

		return dataField;
	}

	public List<?> getClasses(){
		return ClassDictUtil.getArray(this, "classes_");
	}
}