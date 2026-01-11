/*
 * Copyright (c) 2020 Villu Ruusmann
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
package sklearn2pmml.preprocessing;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import scipy.interpolate.BSpline;
import scipy.interpolate.BSplineUtil;
import sklearn.Transformer;

public class BSplineTransformer extends Transformer {

	public BSplineTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		BSpline bspline = getBSpline();

		Feature feature = SchemaUtil.getOnlyFeature(features);

		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		List<DefineFunction> splineFunctions = BSplineUtil.createSplineFunction(bspline, encoder);

		DefineFunction splineFunction = Iterables.getOnlyElement(splineFunctions);

		Apply apply = ExpressionUtil.createApply(splineFunction, continuousFeature.ref());

		DerivedField derivedField = encoder.createDerivedField(createFieldName("bspline", continuousFeature), apply);

		return Collections.singletonList(new ContinuousFeature(encoder, derivedField));
	}

	public BSpline getBSpline(){
		return get("bspline", BSpline.class);
	}
}