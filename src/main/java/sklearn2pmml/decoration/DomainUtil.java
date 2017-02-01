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

import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;

public class DomainUtil {

	private DomainUtil(){
	}

	static
	public MissingValueTreatmentMethod parseMissingValueTreatment(String missingValueTreatment){

		if(missingValueTreatment == null){
			return null;
		}

		switch(missingValueTreatment){
			case "as_is":
				return MissingValueTreatmentMethod.AS_IS;
			case "as_mean":
				return MissingValueTreatmentMethod.AS_MEAN;
			case "as_mode":
				return MissingValueTreatmentMethod.AS_MODE;
			case "as_median":
				return MissingValueTreatmentMethod.AS_MEDIAN;
			case "as_value":
				return MissingValueTreatmentMethod.AS_VALUE;
			default:
				throw new IllegalArgumentException(missingValueTreatment);
		}
	}

	static
	public InvalidValueTreatmentMethod parseInvalidValueTreatment(String invalidValueTreatment){

		if(invalidValueTreatment == null){
			return null;
		}

		switch(invalidValueTreatment){
			case "as_is":
				return InvalidValueTreatmentMethod.AS_IS;
			case "as_missing":
				return InvalidValueTreatmentMethod.AS_MISSING;
			case "return_invalid":
				return InvalidValueTreatmentMethod.RETURN_INVALID;
			default:
				throw new IllegalArgumentException(invalidValueTreatment);
		}
	}
}