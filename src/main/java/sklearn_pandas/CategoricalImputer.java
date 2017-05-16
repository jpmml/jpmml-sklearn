/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn_pandas;

import java.util.Collections;
import java.util.List;

import org.dmg.pmml.MissingValueTreatmentMethod;
import org.jpmml.converter.Feature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn.preprocessing.ImputerUtil;

public class CategoricalImputer extends Transformer {

	public CategoricalImputer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Object fill = getFill();
		Object missingValues = getMissingValues();

		ClassDictUtil.checkSize(1, features);

		if(("NaN").equals(missingValues)){
			missingValues = null;
		}

		Feature feature = features.get(0);

		return Collections.<Feature>singletonList(ImputerUtil.encodeFeature(feature, missingValues, fill, MissingValueTreatmentMethod.AS_MODE, encoder));
	}

	public Object getFill(){
		return get("fill_");
	}

	public Object getMissingValues(){
		return get("missing_values");
	}
}