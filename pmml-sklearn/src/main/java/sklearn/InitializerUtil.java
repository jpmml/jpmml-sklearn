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

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class InitializerUtil {

	private InitializerUtil(){
	}

	static
	public List<Feature> selectFeatures(List<?> columns, List<Feature> features, SkLearnEncoder encoder){
		Function<Object, Feature> castFunction = new Function<Object, Feature>(){

			@Override
			public Feature apply(Object object){

				if(object instanceof String){
					String column = (String)object;

					if(!features.isEmpty()){

						for(Feature feature : features){
							String name = feature.getName();

							if((column).equals(name)){
								return feature;
							}
						}

						throw new IllegalArgumentException("Column \'" + column + "\' is undefined");
					}

					return createWildcardFeature(column);
				} else

				if(object instanceof Integer){
					Integer index = (Integer)object;

					if(!features.isEmpty()){
						Feature feature = features.get(index);

						return feature;
					}

					return createWildcardFeature(("x" + (index.intValue() + 1)));
				} else

				{
					throw new IllegalArgumentException("The column object (" + ClassDictUtil.formatClass(object) + ") is not a string or integer");
				}
			}

			private Feature createWildcardFeature(String name){
				DataField dataField = encoder.getDataField(name);
				if(dataField == null){
					dataField = encoder.createDataField(name);
				}

				return new WildcardFeature(encoder, dataField);
			}
		};

		return Lists.transform(columns, castFunction);
	}
}