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

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.LinearNorm;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutlierTreatmentMethod;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.ClassDictUtil;
import sklearn.Regressor;

public class IsotonicRegression extends Regressor {

	public IsotonicRegression(String module, String name){
		super(module, name);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		List<? extends Number> necessaryX = getNecessaryX();
		List<? extends Number> necessaryY = getNecessaryY();
		String outOfBounds = getOutOfBounds();

		ClassDictUtil.checkSize(necessaryX, necessaryY);

		PMMLEncoder encoder = schema.getEncoder();
		List<? extends Feature> features = schema.getFeatures();

		SchemaUtil.checkSize(1, features);

		Feature feature = features.get(0);

		OutlierTreatmentMethod outlierTreatment = parseOutlierTreatment(outOfBounds);

		NormContinuous normContinuous = new NormContinuous(feature.getName(), null)
			.setOutliers(outlierTreatment);

		for(int i = 0; i < necessaryX.size(); i++){
			Number orig = necessaryX.get(i);
			Number norm = necessaryY.get(i);

			normContinuous.addLinearNorms(new LinearNorm(orig, norm));
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("isotonicRegression", feature), OpType.CONTINUOUS, DataType.DOUBLE, normContinuous);

		feature = new ContinuousFeature(encoder, derivedField);

		return RegressionModelUtil.createRegression(Collections.singletonList(feature), Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.NONE, schema);
	}

	public Boolean getIncreasing(){
		return getBoolean("increasing_");
	}

	public List<? extends Number> getNecessaryX(){
		return getNumberArray("_necessary_X_");
	}

	public List<? extends Number> getNecessaryY(){
		return getNumberArray("_necessary_y_");
	}

	public String getOutOfBounds(){
		return getString("out_of_bounds");
	}

	static
	public OutlierTreatmentMethod parseOutlierTreatment(String outOfBounds){

		switch(outOfBounds){
			case "nan":
				return OutlierTreatmentMethod.AS_MISSING_VALUES;
			case "clip":
				return OutlierTreatmentMethod.AS_EXTREME_VALUES;
			default:
				throw new IllegalArgumentException(outOfBounds);
		}
	}
}