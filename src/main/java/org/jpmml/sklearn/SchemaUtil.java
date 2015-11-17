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
package org.jpmml.sklearn;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.Value;
import org.jpmml.converter.PMMLUtil;

public class SchemaUtil {

	private SchemaUtil(){
	}

	static
	public Schema createSegmentSchema(Schema schema){
		Schema result = new Schema(null, schema.getTargetCategories(), schema.getActiveFields());

		return result;
	}

	static
	public void addValues(DataField dataField, List<?> objects){

		if(objects != null && objects.size() > 0){
			List<Value> values = dataField.getValues();

			Function<Object, String> function = new Function<Object, String>(){

				@Override
				public String apply(Object object){
					return PMMLUtil.formatValue(object);
				}
			};

			values.addAll(PMMLUtil.createValues(Lists.transform(objects, function)));
		}
	}

	static
	public List<OutputField> encodeProbabilityFields(Schema schema){
		List<String> targetCategories = schema.getTargetCategories();

		if(targetCategories != null && targetCategories.size() > 0){
			Function<String, OutputField> function = new Function<String, OutputField>(){

				@Override
				public OutputField apply(String targetCategory){
					return PMMLUtil.createProbabilityField(targetCategory);
				}
			};

			return Lists.transform(targetCategories, function);
		}

		return null;
	}
}