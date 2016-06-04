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
import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;

abstract
public class Classifier extends Estimator {

	public Classifier(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(FeatureMapper featureMapper){

		if(featureMapper.isEmpty()){
			featureMapper.initActiveFields(SchemaUtil.createActiveFields(getNumberOfFeatures()), getOpType(), getDataType());
			featureMapper.initTargetField(SchemaUtil.createTargetField(), OpType.CATEGORICAL, DataType.STRING, getTargetCategories());
		} else

		{
			if(requiresContinuousInput()){
				featureMapper.simplifyActiveFields(true, getOpType(), getDataType());
			}

			featureMapper.updateActiveFields(getNumberOfFeatures(), true, getOpType(), getDataType());
			featureMapper.updateTargetField(OpType.CATEGORICAL, DataType.STRING, getTargetCategories());
		}

		FeatureSchema schema = featureMapper.createSupervisedSchema();

		return encodeModel(schema);
	}

	public boolean hasProbabilityDistribution(){
		return true;
	}

	public List<String> getTargetCategories(){
		List<?> classes = getClasses();

		if(classes == null || classes.isEmpty()){
			throw new IllegalArgumentException();
		}

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

		List<String> targetCategories = new ArrayList<>(Lists.transform(classes, function));

		return targetCategories;
	}

	public List<?> getClasses(){
		return ClassDictUtil.getArray(this, "classes_");
	}
}