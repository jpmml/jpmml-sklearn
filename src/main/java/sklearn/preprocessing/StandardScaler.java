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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import sklearn.Transformer;

public class StandardScaler extends Transformer {

	public StandardScaler(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(String id, List<Feature> inputFeatures, FeatureMapper featureMapper){
		Boolean withMean = getWithMean();
		Boolean withStd = getWithStd();

		List<? extends Number> mean = (withMean ? getMean() : null);
		List<? extends Number> std = (withStd ? getStd() : null);

		if(mean == null && std == null){
			return inputFeatures;
		} // End if

		if(withMean && inputFeatures.size() != mean.size()){
			throw new IllegalArgumentException();
		} // End if

		if(withStd && inputFeatures.size() != std.size()){
			throw new IllegalArgumentException();
		}

		List<Feature> features = new ArrayList<>();

		for(int i = 0; i < inputFeatures.size(); i++){
			Feature inputFeature = inputFeatures.get(i);

			// "($name - mean) / std"
			Expression expression = new FieldRef(inputFeature.getName());

			if(withMean){
				Number meanValue = mean.get(i);

				if(!ValueUtil.isZero(meanValue)){
					expression = PMMLUtil.createApply("-", expression, PMMLUtil.createConstant(meanValue));
				}
			} // End if

			if(withStd){
				Number stdValue = std.get(i);

				if(!ValueUtil.isOne(stdValue)){
					expression = PMMLUtil.createApply("/", expression, PMMLUtil.createConstant(stdValue));
				}
			} // End if

			if(expression instanceof FieldRef){
				features.add(inputFeature);

				continue;
			}

			DerivedField derivedField = featureMapper.createDerivedField(createName(id, i), expression);

			features.add(new ContinuousFeature(derivedField));
		}

		return features;
	}

	public Boolean getWithMean(){
		return (Boolean)get("with_mean");
	}

	public Boolean getWithStd(){
		return (Boolean)get("with_std");
	}

	public List<? extends Number> getMean(){
		return (List)ClassDictUtil.getArray(this, "mean_");
	}

	public List<? extends Number> getStd(){
		try {
			// SkLearn 0.16
			return (List)ClassDictUtil.getArray(this, "std_");
		} catch(IllegalArgumentException iae){
			// SkLearn 0.17
			return (List)ClassDictUtil.getArray(this, "scale_");
		}
	}
}