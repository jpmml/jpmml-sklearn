/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.isotonic;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.LinearNorm;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutlierTreatmentMethod;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Calibrator;

public class IsotonicRegression extends Calibrator {

	public IsotonicRegression(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Number> xThresholds = getXThresholds();
		List<Number> yThresholds = getYThresholds();
		String outOfBounds = getOutOfBounds();

		ClassDictUtil.checkSize(xThresholds, yThresholds);

		SchemaUtil.checkSize(1, features);

		Feature feature = features.get(0);

		OutlierTreatmentMethod outlierTreatment = parseOutlierTreatment(outOfBounds);

		NormContinuous normContinuous = new NormContinuous(feature.getName(), null)
			.setOutlierTreatment(outlierTreatment);

		for(int i = 0; i < xThresholds.size(); i++){
			Number orig = xThresholds.get(i);
			Number norm = yThresholds.get(i);

			normContinuous.addLinearNorms(new LinearNorm(orig, norm));
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("isotonicRegression", feature), OpType.CONTINUOUS, DataType.DOUBLE, normContinuous);

		return Collections.singletonList(new ContinuousFeature(encoder, derivedField));
	}

	public Boolean getIncreasing(){
		return getBoolean("increasing_");
	}

	public List<Number> getXThresholds(){
		// SkLearn 0.23
		if(hasattr("_necessary_X_")){
			return getNumberArray("_necessary_X_");
		}

		// SkLearn 0.24+
		return getNumberArray("X_thresholds_");
	}

	public List<Number> getYThresholds(){
		// SkLearn 0.23
		if(hasattr("_necessary_y_")){
			getNumberArray("_necessary_y_");
		}

		// SkLearn 0.24+
		return getNumberArray("y_thresholds_");
	}

	public String getOutOfBounds(){
		return getEnum("out_of_bounds", this::getString, Arrays.asList(IsotonicRegression.OUTOFBOUNDS_CLIP, IsotonicRegression.OUTOFBOUNDS_NAN));
	}

	static
	public OutlierTreatmentMethod parseOutlierTreatment(String outOfBounds){

		switch(outOfBounds){
			case IsotonicRegression.OUTOFBOUNDS_CLIP:
				return OutlierTreatmentMethod.AS_EXTREME_VALUES;
			case IsotonicRegression.OUTOFBOUNDS_NAN:
				return OutlierTreatmentMethod.AS_MISSING_VALUES;
			default:
				throw new IllegalArgumentException(outOfBounds);
		}
	}

	private static final String OUTOFBOUNDS_CLIP = "clip";
	private static final String OUTOFBOUNDS_NAN = "nan";
}