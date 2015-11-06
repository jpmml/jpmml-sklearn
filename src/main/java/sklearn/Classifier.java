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

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import org.jpmml.sklearn.SchemaUtil;

abstract
public class Classifier extends Estimator {

	public Classifier(String module, String name){
		super(module, name);
	}

	@Override
	public Schema createSchema(){
		FieldName targetField = Schema.createTargetField();
		List<FieldName> activeFields = Schema.createActiveFields(getNumberOfFeatures());

		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				String targetCategory = PMMLUtil.formatValue(object);

				if(targetCategory == null || CharMatcher.WHITESPACE.matchesAnyOf(targetCategory)){
					throw new IllegalArgumentException(targetCategory);
				}

				return targetCategory;
			}
		};

		List<String> targetCategories = new ArrayList<>(Lists.transform(getClasses(), function));
		if(targetCategories.isEmpty()){
			throw new IllegalArgumentException();
		}

		Schema schema = new Schema(targetField, targetCategories, activeFields);

		return schema;
	}

	@Override
	public DataField encodeTargetField(FieldName name, List<String> targetCategories){
		DataField dataField = new DataField(name, OpType.CATEGORICAL, DataType.STRING);

		SchemaUtil.addValues(dataField, targetCategories);

		return dataField;
	}

	public List<?> getClasses(){
		return ClassDictUtil.getArray(this, "classes_");
	}
}