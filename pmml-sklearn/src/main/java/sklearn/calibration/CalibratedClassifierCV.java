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

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import sklearn.Classifier;

public class CalibratedClassifierCV extends Classifier {

	public CalibratedClassifierCV(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<CalibratedClassifier> calibratedClassifiers = getCalibratedClassifiers();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(calibratedClassifiers.size() == 1){
			CalibratedClassifier calibratedClassifier = calibratedClassifiers.get(0);

			return calibratedClassifier.encode(schema);
		} else

		if(calibratedClassifiers.size() >= 2){
			Schema segmentSchema = schema.toAnonymousSchema();

			List<Model> models = new ArrayList<>();

			for(int i = 0; i < calibratedClassifiers.size(); i++){
				CalibratedClassifier calibratedClassifier = calibratedClassifiers.get(i);

				Model model = calibratedClassifier.encode((i + 1), segmentSchema);

				models.add(model);
			}

			MiningModel miningModel = new MiningModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel))
				.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.AVERAGE, Segmentation.MissingPredictionTreatment.RETURN_MISSING, models));

			encodePredictProbaOutput(miningModel, DataType.DOUBLE, categoricalLabel);

			return miningModel;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<CalibratedClassifier> getCalibratedClassifiers(){
		return getList("calibrated_classifiers_", CalibratedClassifier.class);
	}
}