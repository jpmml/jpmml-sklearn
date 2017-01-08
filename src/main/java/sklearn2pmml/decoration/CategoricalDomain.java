/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn2pmml.decoration;

import java.util.Collections;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldDecorator;
import org.jpmml.converter.ValidValueDecorator;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.TypeUtil;

public class CategoricalDomain extends Domain {

	public CategoricalDomain(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		List<?> data = getData();

		return TypeUtil.getDataType(data, DataType.STRING);
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		List<?> data = getData();

		ClassDictUtil.checkSize(1, ids, features);

		final
		InvalidValueTreatmentMethod invalidValueTreatment = DomainUtil.parseInvalidValueTreatment(getInvalidValueTreatment());

		WildcardFeature wildcardFeature = (WildcardFeature)features.get(0);

		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				return ValueUtil.formatValue(object);
			}
		};

		List<String> categories = Lists.transform(data, function);

		FieldDecorator decorator = new ValidValueDecorator(){

			{
				setInvalidValueTreatment(invalidValueTreatment);
			}
		};

		CategoricalFeature categoricalFeature = wildcardFeature.toCategoricalFeature(categories);

		encoder.addDecorator(categoricalFeature.getName(), decorator);

		return Collections.<Feature>singletonList(categoricalFeature);
	}

	public List<?> getData(){
		return (List)ClassDictUtil.getArray(this, "data_");
	}
}