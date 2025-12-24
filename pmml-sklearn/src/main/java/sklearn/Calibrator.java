/*
 * Copyright (c) 2023 Villu Ruusmann
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

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn2pmml.HasPMMLName;

abstract
public class Calibrator extends SkLearnRegressor implements HasPMMLName<Calibrator> {

	public Calibrator(String module, String name){
		super(module, name);
	}

	/**
	 * @see Transformer#encodeFeatures(List, SkLearnEncoder)
	 */
	abstract
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder);

	@Override
	public RegressionModel encodeModel(Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		List<? extends Feature> features = schema.getFeatures();

		features = encodeFeatures((List)features, encoder);

		Feature feature = Iterables.getOnlyElement(features);

		return RegressionModelUtil.createRegression(Collections.singletonList(feature), Collections.singletonList(1d), 0d, RegressionModel.NormalizationMethod.NONE, schema);
	}

	/**
	 * @see Transformer#createFieldName(String, Object...)
	 */
	public String createFieldName(String function, Feature feature){
		String pmmlName = getPMMLName();

		if(pmmlName != null){
			return pmmlName;
		}

		return FieldNameUtil.create(function, Collections.singletonList(feature));
	}

	@Override
	public Calibrator setPMMLName(String pmmlName){
		return (Calibrator)super.setPMMLName(pmmlName);
	}
}