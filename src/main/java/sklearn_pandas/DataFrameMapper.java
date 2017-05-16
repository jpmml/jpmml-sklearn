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
package sklearn_pandas;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TupleUtil;
import sklearn.Initializer;
import sklearn.Transformer;

public class DataFrameMapper extends Initializer {

	public DataFrameMapper(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		Object _default = getDefault();
		List<Object[]> rows = getFeatures();

		if(!(Boolean.FALSE).equals(_default)){
			throw new IllegalArgumentException();
		}

		List<Feature> result = new ArrayList<>();

		for(Object[] row : rows){
			List<Feature> rowFeatures = new ArrayList<>();

			List<String> columns = getColumnList(row);
			for(String column : columns){
				FieldName name = FieldName.create(column);

				DataField dataField = encoder.getDataField(name);
				if(dataField == null){
					dataField = encoder.createDataField(name);
				}

				rowFeatures.add(new WildcardFeature(encoder, dataField));
			}

			List<Transformer> transformers = getTransformerList(row);
			for(Transformer transformer : transformers){
				encoder.updateFeatures(rowFeatures, transformer);

				rowFeatures = transformer.encodeFeatures(rowFeatures, encoder);
			}

			if(row.length > 2){
				Map<String, ?> options = (Map)row[2];

				String alias = (String)options.get("alias");
				if(alias != null){

					for(int i = 0; i < rowFeatures.size(); i++){
						Feature rowFeature = rowFeatures.get(i);

						encoder.renameField(rowFeature.getName(), rowFeatures.size() > 1 ? FieldName.create(alias + "_" + i) : FieldName.create(alias));
					}
				}
			}

			result.addAll(rowFeatures);
		}

		return result;
	}

	public Object getDefault(){
		return get("default");
	}

	public DataFrameMapper setDefault(Object _default){
		put("default", _default);

		return this;
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}

	public DataFrameMapper setFeatures(List<Object[]> features){
		put("features", features);

		return this;
	}

	static
	private List<String> getColumnList(Object[] feature){
		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){

				if(object instanceof String){
					return (String)object;
				}

				throw new IllegalArgumentException("The key object (" + ClassDictUtil.formatClass(object) + ") is not a String");
			}
		};

		try {
			if(feature[0] instanceof HasArray){
				HasArray hasArray = (HasArray)feature[0];

				return (List)hasArray.getArrayContent();
			} // End if

			if(feature[0] instanceof List){
				return Lists.transform(((List)feature[0]), function);
			}

			return Collections.singletonList(function.apply(feature[0]));
		} catch(RuntimeException re){
			throw new IllegalArgumentException("Invalid mapping key", re);
		}
	}

	static
	private List<Transformer> getTransformerList(Object[] feature){
		Function<Object, Transformer> function = new Function<Object, Transformer>(){

			@Override
			public Transformer apply(Object object){

				if(object instanceof Transformer){
					return (Transformer)object;
				}

				throw new IllegalArgumentException("The value object (" + ClassDictUtil.formatClass(object) + ") is not a Transformer or is not a supported Transformer subclass");
			}
		};

		try {
			if(feature[1] == null){
				return Collections.emptyList();
			} // End if

			if(feature[1] instanceof TransformerPipeline){
				TransformerPipeline transformerPipeline = (TransformerPipeline)feature[1];

				List<Object[]> steps = transformerPipeline.getSteps();

				return Lists.transform((List)TupleUtil.extractElementList(steps, 1), function);
			} // End if

			if(feature[1] instanceof List){
				return Lists.transform((List)feature[1], function);
			}

			return Collections.singletonList(function.apply(feature[1]));
		} catch(RuntimeException re){
			throw new IllegalArgumentException("Invalid mapping value", re);
		}
	}
}