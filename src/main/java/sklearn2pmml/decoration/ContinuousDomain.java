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

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Interval;
import org.dmg.pmml.InvalidValueTreatmentMethodType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PseudoFeature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import org.jpmml.sklearn.FieldDecorator;
import org.jpmml.sklearn.ValidValueDecorator;

public class ContinuousDomain extends Domain {

	public ContinuousDomain(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> inputFeatures, FeatureMapper featureMapper){
		List<? extends Number> dataMin = getDataMin();
		List<? extends Number> dataMax = getDataMax();

		if(ids.size() != inputFeatures.size() || dataMin.size() != inputFeatures.size() || dataMax.size() != inputFeatures.size()){
			throw new IllegalArgumentException();
		}

		final
		InvalidValueTreatmentMethodType invalidValueTreatment = DomainUtil.parseInvalidValueTreatment(getInvalidValueTreatment());

		for(int i = 0; i < inputFeatures.size(); i++){
			PseudoFeature inputFeature = (PseudoFeature)inputFeatures.get(i);

			final
			Interval interval = new Interval(Interval.Closure.CLOSED_CLOSED)
				.setLeftMargin(ValueUtil.asDouble(dataMin.get(i)))
				.setRightMargin(ValueUtil.asDouble(dataMax.get(i)));

			FieldDecorator decorator = new ValidValueDecorator(){

				{
					setInvalidValueTreatment(invalidValueTreatment);
				}

				@Override
				public void decorate(DataField dataField, MiningField miningField){
					DataType dataType = dataField.getDataType();

					switch(dataType){
						case FLOAT:
							{
								interval.setLeftMargin(toFloat(interval.getLeftMargin()));
								interval.setRightMargin(toFloat(interval.getRightMargin()));
							}
							break;
						default:
							break;
					}

					dataField.addIntervals(interval);

					super.decorate(dataField, miningField);
				}

				private Double toFloat(Double value){

					if((value != null) && (value.doubleValue() != value.floatValue())){
						return (double)value.floatValue();
					}

					return value;
				}
			};

			featureMapper.addDecorator(inputFeature.getName(), decorator);
		}

		return inputFeatures;
	}

	public List<? extends Number> getDataMin(){
		return (List)ClassDictUtil.getArray(this, "data_min_");
	}

	public List<? extends Number> getDataMax(){
		return (List)ClassDictUtil.getArray(this, "data_max_");
	}
}