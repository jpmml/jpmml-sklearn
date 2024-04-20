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
import java.util.Objects;

import com.google.common.collect.Lists;
import org.jpmml.converter.Feature;
import org.jpmml.python.AttributeException;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Initializer;
import sklearn.InitializerUtil;
import sklearn.Transformer;

public class DataFrameMapper extends Initializer {

	public DataFrameMapper(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		return encodeFeatures(Collections.emptyList(), encoder);
	}

	@SuppressWarnings("unchecked")
	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		@SuppressWarnings("unused")
		Boolean _default = getDefault();
		List<Object[]> rows = getFeatures();

		List<Feature> result = new ArrayList<>();

		for(Object[] row : rows){
			List<String> columns = getColumnList(row);
			List<Transformer> transformers = getTransformerList(row);

			List<Feature> rowFeatures = InitializerUtil.selectFeatures(columns, features, encoder);

			for(Transformer transformer : transformers){
				rowFeatures = transformer.encode(rowFeatures, encoder);
			}

			if(row.length > 2){
				Map<String, ?> options = (Map<String, ?>)row[2];

				String alias = (String)options.get("alias");
				if(alias != null){

					for(int i = 0; i < rowFeatures.size(); i++){
						Feature rowFeature = rowFeatures.get(i);

						encoder.renameFeature(rowFeature, rowFeatures.size() > 1 ? (alias + "_" + i) : alias);
					}
				}
			}

			result.addAll(rowFeatures);
		}

		return result;
	}

	public Boolean getDefault(){
		Object object = getOptionalObject("default");

		if(!Objects.equals(Boolean.FALSE, object)){
			throw new AttributeException("Attribute \'" + ClassDictUtil.formatMember(this, "default") + "\' must be set to the 'False' value");
		}

		return (Boolean)object;
	}

	public DataFrameMapper setDefault(Object _default){
		setattr("default", _default);

		return this;
	}

	public List<Object[]> getFeatures(){
		return getTupleList("features");
	}

	public DataFrameMapper setFeatures(List<Object[]> features){
		setattr("features", features);

		return this;
	}

	static
	private List<String> getColumnList(Object[] feature){
		Object key = feature[0];

		if(key instanceof HasArray){
			HasArray hasArray = (HasArray)key;

			key = hasArray.getArrayContent();
		}

		CastFunction<String> castFunction = new CastFunction<String>(String.class){

			@Override
			protected String formatMessage(Object object){
				return "The key object (" + ClassDictUtil.formatClass(object) + ") is not a String";
			}
		};

		if(key instanceof List){
			return Lists.transform((List<?>)key, castFunction);
		}

		return Collections.singletonList(castFunction.apply(key));
	}

	static
	private List<Transformer> getTransformerList(Object[] feature){
		Object value = feature[1];

		if(value == null){
			return Collections.emptyList();
		} // End if

		if(value instanceof TransformerPipeline){
			TransformerPipeline transformerPipeline = (TransformerPipeline)value;

			List<Object[]> steps = transformerPipeline.getSteps();

			value = TupleUtil.extractElementList(steps, 1);
		}

		CastFunction<Transformer> castFunction = new CastFunction<Transformer>(Transformer.class){

			@Override
			protected String formatMessage(Object object){
				return "The value object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer";
			}
		};

		if(value instanceof List){
			return Lists.transform((List<?>)value, castFunction);
		}

		return Collections.singletonList(castFunction.apply(value));
	}
}