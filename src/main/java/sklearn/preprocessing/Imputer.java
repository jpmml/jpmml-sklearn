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

import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.TypeDefinitionField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class Imputer extends Transformer {

	public Imputer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<String> ids, List<Feature> features, SkLearnEncoder encoder){
		List<? extends Number> statistics = getStatistics();

		ClassDictUtil.checkSize(ids, features, statistics);

		Object missingValues = getMissingValues();

		Number targetValue = getTargetValue(missingValues);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			String id = ids.get(i);
			Feature feature = features.get(i);

			Number statisticValue = statistics.get(i);

			TypeDefinitionField field = encoder.getField(feature.getName());
			if(field instanceof DataField){
				String strategy = getStrategy();

				MissingValueDecorator decorator = new MissingValueDecorator()
					.setMissingValueReplacement(ValueUtil.formatValue(statisticValue))
					.setMissingValueTreatment(parseStrategy(strategy));

				if(targetValue != null){
					decorator.addMissingValues(ValueUtil.formatValue(targetValue));
				}

				encoder.addDecorator(feature.getName(), decorator);

				result.add(feature);
			} else

			{
				Expression expression = feature.ref();

				if(targetValue == null){
					expression = PMMLUtil.createApply("isMissing", expression);
				} else

				{
					expression = PMMLUtil.createApply("equal", expression, PMMLUtil.createConstant(targetValue));
				}

				// "($name == null) ? statistics : $name"
				expression = PMMLUtil.createApply("if", expression, PMMLUtil.createConstant(statisticValue), feature.ref());

				DerivedField derivedField = encoder.createDerivedField(createName(id), expression);

				result.add(new ContinuousFeature(encoder, derivedField));
			}
		}

		return result;
	}

	public Object getMissingValues(){
		return get("missing_values");
	}

	public List<? extends Number> getStatistics(){
		return (List)ClassDictUtil.getArray(this, "statistics_");
	}

	public String getStrategy(){
		return (String)get("strategy");
	}

	static
	private Number getTargetValue(Object object){

		if(object instanceof String){
			return null;
		} else

		if(object instanceof Number){
			return (Number)object;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	private MissingValueTreatmentMethod parseStrategy(String strategy){

		switch(strategy){
			case "mean":
				return MissingValueTreatmentMethod.AS_MEAN;
			case "median":
				return MissingValueTreatmentMethod.AS_MEDIAN;
			case "most_frequent":
				return MissingValueTreatmentMethod.AS_MODE;
			default:
				throw new IllegalArgumentException(strategy);
		}
	}
}