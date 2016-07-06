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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;

abstract
public class Classifier extends Estimator {

	public Classifier(String module, String name){
		super(module, name);
	}

	@Override
	public Schema createSchema(FeatureMapper featureMapper){
		List<?> classes = getClasses();

		if(classes == null || classes.isEmpty()){
			throw new IllegalArgumentException();
		}

		DataType dataType = getDataType(classes);

		List<String> targetCategories = formatTargetCategories(classes);

		if(featureMapper.isEmpty()){
			featureMapper.initActiveFields(createActiveFields(getNumberOfFeatures()), getOpType(), getDataType());
			featureMapper.initTargetField(createTargetField(), OpType.CATEGORICAL, dataType, targetCategories);
		} else

		{
			featureMapper.updateActiveFields(getNumberOfFeatures(), true, getOpType(), getDataType());
			featureMapper.updateTargetField(OpType.CATEGORICAL, dataType, targetCategories);
		}

		Schema schema = featureMapper.createSupervisedSchema();

		if(requiresContinuousInput()){
			schema = featureMapper.cast(OpType.CONTINUOUS, getDataType(), schema);
		}

		return schema;
	}

	public boolean hasProbabilityDistribution(){
		return true;
	}

	public List<?> getClasses(){
		return ClassDictUtil.getArray(this, "classes_");
	}

	static
	private DataType getDataType(List<?> objects){
		Function<Object, Class<?>> function = new Function<Object, Class<?>>(){

			@Override
			public Class<?> apply(Object object){
				return object.getClass();
			}
		};

		Set<Class<?>> clazzes = new HashSet<>(Lists.transform(objects, function));

		Class<?> clazz = Iterables.getOnlyElement(clazzes);

		DataType dataType = Classifier.dataTypes.get(clazz);
		if(dataType != null){
			return dataType;
		}

		return DataType.STRING;
	}

	static
	private List<String> formatTargetCategories(List<?> objects){
		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				String targetCategory = ValueUtil.formatValue(object);

				if(targetCategory == null || CharMatcher.WHITESPACE.matchesAnyOf(targetCategory)){
					throw new IllegalArgumentException(targetCategory);
				}

				return targetCategory;
			}
		};

		return new ArrayList<>(Lists.transform(objects, function));
	}

	private static final Map<Class<?>, DataType> dataTypes = new LinkedHashMap<>();

	static {
		dataTypes.put(Boolean.class, DataType.BOOLEAN);
		dataTypes.put(Byte.class, DataType.INTEGER);
		dataTypes.put(Short.class, DataType.INTEGER);
		dataTypes.put(Integer.class, DataType.INTEGER);
		dataTypes.put(Long.class, DataType.INTEGER);
		dataTypes.put(Float.class, DataType.FLOAT);
		dataTypes.put(Double.class, DataType.DOUBLE);
	}
}