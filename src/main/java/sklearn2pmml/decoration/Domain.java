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

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.Counts;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

abstract
public class Domain extends Transformer {

	public Domain(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		MissingValueTreatmentMethod missingValueTreatment = DomainUtil.parseMissingValueTreatment(getMissingValueTreatment());
		Object missingValueReplacement = getMissingValueReplacement();

		if(missingValueReplacement != null){

			if(missingValueTreatment == null){
				missingValueTreatment = MissingValueTreatmentMethod.AS_VALUE;
			}
		}

		InvalidValueTreatmentMethod invalidValueTreatment = DomainUtil.parseInvalidValueTreatment(getInvalidValueTreatment());

		for(Feature feature : features){
			FieldName name = feature.getName();

			if(missingValueTreatment != null){
				MissingValueDecorator missingValueDecorator = new MissingValueDecorator()
					.setMissingValueTreatment(missingValueTreatment)
					.setMissingValueReplacement(missingValueReplacement != null ? ValueUtil.formatValue(missingValueReplacement) : null);

				encoder.addDecorator(name, missingValueDecorator);
			} // End if

			if(invalidValueTreatment != null){
				InvalidValueDecorator invalidValueDecorator = new InvalidValueDecorator()
					.setInvalidValueTreatment(invalidValueTreatment);

				encoder.addDecorator(name, invalidValueDecorator);
			}
		}

		return features;
	}

	public String getMissingValueTreatment(){
		return (String)get("missing_value_treatment");
	}

	public Object getMissingValueReplacement(){
		return get("missing_value_replacement");
	}

	public String getInvalidValueTreatment(){
		return (String)get("invalid_value_treatment");
	}

	public Boolean getWithData(){
		Boolean withData = (Boolean)get("with_data");

		if(withData == null){
			return Boolean.TRUE;
		}

		return withData;
	}

	public Boolean getWithStatistics(){
		Boolean withStatistics = (Boolean)get("with_statistics");

		if(withStatistics == null){
			return Boolean.FALSE;
		}

		return withStatistics;
	}

	public Map<String, ?> getCounts(){
		return (Map)get("counts_");
	}

	static
	public Counts createCounts(Map<String, ?> values){
		Counts counts = new Counts()
			.setTotalFreq(selectValue(values, "totalFreq", 0d))
			.setMissingFreq(selectValue(values, "missingFreq"))
			.setInvalidFreq(selectValue(values, "invalidFreq"));

		return counts;
	}

	static
	protected Map<String, ?> extractMap(Map<String, ?> map, int index){
		Map<String, Object> result = new LinkedHashMap<>();

		Collection<? extends Map.Entry<String, ?>> entries = map.entrySet();
		for(Map.Entry<String, ?> entry : entries){
			String key = entry.getKey();
			List<?> value = asArray(entry.getValue());

			result.put(key, value.get(index));
		}

		return result;
	}

	static
	protected Double selectValue(Map<String, ?> values, String key){
		return selectValue(values, key, null);
	}

	static
	protected Double selectValue(Map<String, ?> values, String key, Double defaultValue){
		Number value = (Number)values.get(key);

		if(value == null){
			return defaultValue;
		}

		return ValueUtil.asDouble(value);
	}

	static
	protected List<?> asArray(Object object){

		if(object instanceof HasArray){
			HasArray hasArray = (HasArray)object;

			return hasArray.getArrayContent();
		} // End if

		if(object instanceof Number){
			return Collections.singletonList(object);
		}

		throw new IllegalArgumentException(ClassDictUtil.formatClass(object) + " is not a supported array type");
	}
}