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
package sklearn.linear_model.ridge;

import java.util.List;

import sklearn.linear_model.LinearClassifier;
import sklearn.preprocessing.LabelBinarizer;

public class RidgeClassifier extends LinearClassifier {

	public RidgeClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		LabelBinarizer labelBinarizer = getLabelBinarizer();

		return labelBinarizer.getClasses();
	}

	@Override
	public boolean hasProbabilityDistribution(){
		return false;
	}

	@Override
	public int[] getCoefShape(){
		int[] shape = getArrayShape("coef_");

		// SkLearn 1.6.0+
		if(shape.length == 1){
			return new int[]{1, shape[0]};
		}

		return super.getCoefShape();
	}

	public LabelBinarizer getLabelBinarizer(){
		return get("_label_binarizer", LabelBinarizer.class);
	}
}