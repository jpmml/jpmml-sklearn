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
package sklearn.linear_model;

import java.util.List;

import org.dmg.pmml.MiningModel;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import sklearn.preprocessing.LabelBinarizer;

public class RidgeClassifier extends BaseLinearClassifier {

	public RidgeClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		LabelBinarizer labelBinarizer = getLabelBinarizer();

		return labelBinarizer.getClasses();
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		MiningModel miningModel = super.encodeModel(schema)
			.setOutput(null);

		return miningModel;
	}

	public LabelBinarizer getLabelBinarizer(){
		Object object = get("_label_binarizer");

		try {
			if(object == null){
				throw new NullPointerException();
			}

			return (LabelBinarizer)object;
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The label binarizer object (" + ClassDictUtil.formatClass(object) + ") is not a LabelBinarizer", re);
		}
	}
}