/*
 * Copyright (c) 2024 Villu Ruusmann
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

import org.dmg.pmml.Apply;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import scipy.interpolate.BSpline;
import scipy.interpolate.BSplineUtil;
import sklearn.SkLearnTransformer;

public class SplineTransformer extends SkLearnTransformer {

	public SplineTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<BSpline> bsplines = getBSplines();
		String extrapolation = getExtrapolation();
		Boolean includeBias = getIncludeBias();

		switch(extrapolation){
			case "error":
				break;
			default:
				throw new IllegalArgumentException(extrapolation);
		}

		if(!includeBias){
			throw new IllegalArgumentException();
		}

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		BSpline bspline = bsplines.get(0);

		List<DefineFunction> splineFunctions = BSplineUtil.createSplineFunction(bspline, encoder);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < splineFunctions.size(); i++){
			DefineFunction splineFunction = splineFunctions.get(i);

			Apply apply = ExpressionUtil.createApply(splineFunction, continuousFeature.ref());

			DerivedField derivedField = encoder.createDerivedField(createFieldName("bspline", continuousFeature, i), apply);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public List<BSpline> getBSplines(){
		return getList("bsplines_", BSpline.class);
	}

	public String getExtrapolation(){
		return getString("extrapolation");
	}

	public Boolean getIncludeBias(){
		return getBoolean("include_bias");
	}
}