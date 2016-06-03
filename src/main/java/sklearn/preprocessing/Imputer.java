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
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import sklearn.Transformer;

public class Imputer extends Transformer {

	public Imputer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(String id, List<Feature> inputFeatures, FeatureMapper featureMapper){
		List<? extends Number> statistics = getStatistics();

		if(statistics.size() != inputFeatures.size()){
			throw new IllegalArgumentException();
		}

		Object missingValues = getMissingValues();

		List<Feature> features = new ArrayList<>();

		for(int i = 0; i < inputFeatures.size(); i++){
			Feature inputFeature = inputFeatures.get(i);

			Expression expression = new FieldRef(inputFeature.getName());

			if(missingValues instanceof String){
				expression = PMMLUtil.createApply("isMissing", expression);
			} else

			if(missingValues instanceof Number){
				Number number = (Number)missingValues;

				expression = PMMLUtil.createApply("equal", expression, PMMLUtil.createConstant(number));
			} else

			{
				throw new IllegalArgumentException();
			}

			Number statisticValue = statistics.get(i);

			// "($name == null) ? statistics : $name"
			expression = PMMLUtil.createApply("if", expression, PMMLUtil.createConstant(statisticValue), new FieldRef(inputFeature.getName()));

			DerivedField derivedField = featureMapper.createDerivedField(createName(id, i), expression);

			features.add(new ContinuousFeature(derivedField));
		}

		return features;
	}

	public Object getMissingValues(){
		return get("missing_values");
	}

	public List<? extends Number> getStatistics(){
		return (List)ClassDictUtil.getArray(this, "statistics_");
	}
}