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

import java.util.List;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.ValueUtil;
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
}