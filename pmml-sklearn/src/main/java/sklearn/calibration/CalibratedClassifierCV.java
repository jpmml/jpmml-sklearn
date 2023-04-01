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
package sklearn.calibration;

import java.util.List;

import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import sklearn.Classifier;

public class CalibratedClassifierCV extends Classifier {

	public CalibratedClassifierCV(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<CalibratedClassifier> calibratedClassifiers = getCalibratedClassifiers();

		if(calibratedClassifiers.size() == 1){
			CalibratedClassifier calibratedClassifier = calibratedClassifiers.get(0);

			return calibratedClassifier.encode(schema);
		} else

		if(calibratedClassifiers.size() >= 2){
			throw new IllegalArgumentException();
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<CalibratedClassifier> getCalibratedClassifiers(){
		return getList("calibrated_classifiers_", CalibratedClassifier.class);
	}
}