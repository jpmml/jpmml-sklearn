/*
 * Copyright (c) 2022 Villu Ruusmann
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

import java.util.Collections;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonException;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;

public class InitializerUtil {

	private InitializerUtil(){
	}

	static
	public Feature selectFeature(String column, List<Feature> features, SkLearnEncoder encoder){
		List<Feature> result = selectFeatures(Collections.singletonList(column), features, encoder);

		return Iterables.getOnlyElement(result);
	}

	static
	public List<Feature> selectFeatures(List<?> columns, List<Feature> features, SkLearnEncoder encoder){
		Function<Object, Feature> castFunction = new Function<Object, Feature>(){

			@Override
			public Feature apply(Object object){

				if(object instanceof String){
					String column = (String)object;

					if(!features.isEmpty()){
						Feature feature = FeatureUtil.findFeature(features, column);

						if(feature != null){
							return feature;
						}

						throw new SkLearnException("Column \'" + column + "\' not found in " + FeatureUtil.formatNames(features, '\''));
					}

					return createWildcardFeature(column, encoder);
				} else

				if(object instanceof Integer){
					Integer index = (Integer)object;

					if(!features.isEmpty()){
						Feature feature = features.get(index);

						return feature;
					}

					return createWildcardFeature(("x" + (index.intValue() + 1)), encoder);
				} else

				{
					throw new PythonException("The column object (" + ClassDictUtil.formatClass(object) + ") is not a string nor integer");
				}
			}
		};

		return Lists.transform(columns, castFunction);
	}

	static
	public Feature createWildcardFeature(String name, SkLearnEncoder encoder){
		DataField dataField = encoder.getDataField(name);
		if(dataField == null){
			dataField = encoder.createDataField(name);
		}

		return new WildcardFeature(encoder, dataField);
	}
}