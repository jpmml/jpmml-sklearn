/*
 * Copyright (c) 2020 Villu Ruusmann
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
package category_encoders;

import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import numpy.DType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.python.PythonObject;
import sklearn.Transformer;

abstract
public class CategoryEncoder extends Transformer {

	public CategoryEncoder(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		return DataType.STRING;
	}

	public List<String> getDropCols(){
		return getList("drop_cols", String.class);
	}

	public Boolean getDropInvariant(){
		return getBoolean("drop_invariant");
	}

	public String getHandleMissing(){
		return getString("handle_missing");
	}

	public String getHandleUnknown(){
		return getString("handle_unknown");
	}

	public List<Mapping> getMapping(){
		List<Map<String, ?>> mapping = (List)getList("mapping", Map.class);

		Function<Map<String, ?>, Mapping> function = new Function<Map<String, ?>, Mapping>(){

			@Override
			public Mapping apply(Map<String, ?> map){
				Mapping mapping = CategoryEncoder.this.new Mapping("mapping");
				mapping.putAll(map);

				return mapping;
			}
		};

		return Lists.transform(mapping, function);
	}

	public class Mapping extends PythonObject {

		private Mapping(String name){
			super(CategoryEncoder.this.getPythonModule() + "." + CategoryEncoder.this.getPythonName(), name);
		}

		public Integer getCol(){
			return getInteger("col");
		}

		public DType getDataType(){
			return get("data_type", DType.class);
		}

		public <E extends PythonObject> E getMapping(Class<? extends E> clazz){
			return get("mapping", clazz);
		}
	}

	public static final Object CATEGORY_MISSING = Double.NaN;
}