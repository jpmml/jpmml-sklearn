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

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.DataField;
import org.dmg.pmml.InvalidValueTreatmentMethodType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.PseudoFeature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import org.jpmml.sklearn.FieldDecorator;
import org.jpmml.sklearn.ValidValueDecorator;

public class CategoricalDomain extends Domain {

	public CategoricalDomain(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> inputFeatures, FeatureMapper featureMapper){
		List<?> data = getData();

		if(ids.size() != 1 || inputFeatures.size() != 1){
			throw new IllegalArgumentException();
		}

		final
		InvalidValueTreatmentMethodType invalidValueTreatment = DomainUtil.parseInvalidValueTreatment(getInvalidValueTreatment());

		PseudoFeature inputFeature = (PseudoFeature)inputFeatures.get(0);

		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object object){
				return ValueUtil.formatValue(object);
			}
		};

		final
		List<String> categories = Lists.transform(data, function);

		FieldDecorator decorator = new ValidValueDecorator(){

			{
				setInvalidValueTreatment(invalidValueTreatment);
			}

			@Override
			public void decorate(DataField dataField, MiningField miningField){
				List<Value> values = dataField.getValues();

				if(categories.size() > 0){
					values.addAll(PMMLUtil.createValues(categories));
				}

				super.decorate(dataField, miningField);
			}
		};

		featureMapper.addDecorator(inputFeature.getName(), decorator);

		return inputFeatures;
	}

	public List<?> getData(){
		return (List)ClassDictUtil.getArray(this, "data_");
	}
}