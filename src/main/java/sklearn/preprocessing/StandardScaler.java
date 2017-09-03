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
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasNumberOfFeatures;
import sklearn.Transformer;

public class StandardScaler extends Transformer implements HasNumberOfFeatures {

	public StandardScaler(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		Boolean withMean = getWithMean();
		Boolean withStd = getWithStd();

		int[] shape;

		if(withMean){
			shape = getMeanShape();
		} else

		if(withStd){
			shape = getStdShape();
		} else

		{
			return -1;
		}

		return shape[0];
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Boolean withMean = getWithMean();
		Boolean withStd = getWithStd();

		List<? extends Number> mean = (withMean ? getMean() : null);
		List<? extends Number> std = (withStd ? getStd() : null);

		if(mean == null && std == null){
			return features;
		}

		ClassDictUtil.checkSize(features, mean, std);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			Number meanValue = (withMean ? mean.get(i) : 0d);
			Number stdValue = (withStd ? std.get(i) : 1d);

			if(ValueUtil.isZero(meanValue) && ValueUtil.isOne(stdValue)){
				result.add(feature);

				continue;
			}

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			// "($name - mean) / std"
			Expression expression = continuousFeature.ref();

			if(!ValueUtil.isZero(meanValue)){
				expression = PMMLUtil.createApply("-", expression, PMMLUtil.createConstant(meanValue));
			} // End if

			if(!ValueUtil.isOne(stdValue)){
				expression = PMMLUtil.createApply("/", expression, PMMLUtil.createConstant(stdValue));
			}

			DerivedField derivedField = encoder.createDerivedField(createName(continuousFeature), expression);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
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

		// SkLearn 0.16
		if(containsKey("std_")){
			return (List)ClassDictUtil.getArray(this, "std_");
		}

		// SkLearn 0.17+
		return (List)ClassDictUtil.getArray(this, "scale_");
	}

	private int[] getMeanShape(){
		return ClassDictUtil.getShape(this, "mean_", 1);
	}

	private int[] getStdShape(){

		// SkLearn 0.16
		if(containsKey("std_")){
			return ClassDictUtil.getShape(this, "std_", 1);
		}

		// SkLearn 0.17+
		return ClassDictUtil.getShape(this, "scale_", 1);
	}
}