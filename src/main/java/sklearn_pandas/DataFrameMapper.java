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

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TupleUtil;
import sklearn.Transformer;

public class DataFrameMapper extends ClassDict {

	public DataFrameMapper(String module, String name){
		super(module, name);
	}

	public void encodeFeatures(SkLearnEncoder encoder){
		List<Object[]> steps = getFeatures();

		for(int row = 0; row < steps.size(); row++){
			Object[] step = steps.get(row);

			List<String> ids = new ArrayList<>();
			List<Feature> features = new ArrayList<>();

			List<String> names = getNameList(step);
			for(String name : names){
				ids.add(name);

				DataField dataField = encoder.createDataField(FieldName.create(name));

				Feature feature = new WildcardFeature(encoder, dataField);

				features.add(feature);
			}

			List<Transformer> transformers = getTransformerList(step);
			for(int column = 0; column < transformers.size(); column++){
				Transformer transformer = transformers.get(column);

				for(Feature feature : features){
					encoder.updateType(feature.getName(), transformer.getOpType(), transformer.getDataType());
				}

				features = transformer.encodeFeatures(ids, features, encoder);
			}

			encoder.addRow(ids, features);
		}
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}

	static
	private List<String> getNameList(Object[] feature){
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
			if(feature[0] instanceof List){
				return new ArrayList<>(Lists.transform(((List)feature[0]), function));
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

				return new ArrayList<>(Lists.transform((List)TupleUtil.extractElementList(steps, 1), function));
			} // End if

			if(feature[1] instanceof List){
				return new ArrayList<>(Lists.transform((List)feature[1], function));
			}

			return Collections.singletonList(function.apply(feature[1]));
		} catch(RuntimeException re){
			throw new IllegalArgumentException("Invalid mapping value", re);
		}
	}
}