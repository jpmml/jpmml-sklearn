/*
 * Copyright (c) 2026 Villu Ruusmann
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
package causalml.meta;

import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Target;
import org.dmg.pmml.Targets;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;

public class BaseLearnerUtil {

	private BaseLearnerUtil(){
	}

	static
	public MiningModel encodeBinaryModel(Model controlModel, Model treatmentModel, Schema schema){
		Targets targets = controlModel.getTargets();

		if(targets != null){
			Target target = Iterables.getOnlyElement(targets);

			Number rescaleFactor = target.getRescaleFactor();
			Number rescaleConstant = target.getRescaleConstant();

			if(rescaleFactor.doubleValue() != 0d){
				target.setRescaleFactor((Number)ValueUtil.toNegative(rescaleFactor));
			} // End if

			if(rescaleConstant.doubleValue() != 0d){
				target.setRescaleConstant((Number)ValueUtil.toNegative(rescaleConstant));
			}
		} else

		{
			ContinuousLabel continuousLabel = new ContinuousLabel(null, DataType.DOUBLE);

			targets = ModelUtil.createRescaleTargets(-1, null, continuousLabel);

			controlModel.setTargets(targets);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethod.SUM, Segmentation.MissingPredictionTreatment.RETURN_MISSING, Arrays.asList(treatmentModel, controlModel)));

		return miningModel;
	}

	static
	public Model encodeModel(List<Model> models){

		if(models.size() == 1){
			Model model = Iterables.getOnlyElement(models);

			return model;
		} else

		{
			return MiningModelUtil.createMultiModelChain(models, Segmentation.MissingPredictionTreatment.RETURN_MISSING);
		}
	}
}