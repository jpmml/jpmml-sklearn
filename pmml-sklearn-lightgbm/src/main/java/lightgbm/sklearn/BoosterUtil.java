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
package lightgbm.sklearn;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.lightgbm.GBDT;
import org.jpmml.lightgbm.HasLightGBMOptions;
import org.jpmml.lightgbm.ObjectiveFunction;
import sklearn.Estimator;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public <E extends Estimator & HasBooster> int getNumberOfFeatures(E estimator){
		GBDT gbdt = getGBDT(estimator);

		String[] featureNames = gbdt.getFeatureNames();

		return featureNames.length;
	}

	static
	public <E extends Estimator & HasBooster> ObjectiveFunction getObjectiveFunction(E estimator){
		GBDT gbdt = getGBDT(estimator);

		return gbdt.getObjectiveFunction();
	}

	static
	public <E extends Estimator & HasBooster & HasLightGBMOptions> Model encodeModel(E estimator, Schema schema){
		GBDT gbdt = getGBDT(estimator);

		ModelEncoder encoder = schema.getEncoder();

		Integer numIteration = getNumIteration(estimator);

		Model model = gbdt.encodeModel(numIteration, schema);

		encoder.transferFeatureImportances(null, model);

		return model;
	}

	static
	public <E extends Estimator & HasBooster & HasLightGBMOptions> Schema configureSchema(E estimator, Schema schema){
		GBDT gbdt = getGBDT(estimator);

		Map<String, ?> options = getOptions(estimator);

		Schema lgbmSchema = gbdt.toLightGBMSchema(schema);

		lgbmSchema = gbdt.configureSchema(options, lgbmSchema);

		return lgbmSchema;
	}

	static
	public <E extends Estimator & HasBooster & HasLightGBMOptions> Model configureModel(E estimator, Model model){
		GBDT gbdt = getGBDT(estimator);

		Map<String, ?> options = getOptions(estimator);

		model = gbdt.configureModel(options, model);

		return model;
	}

	static
	public <E extends Estimator & HasBooster & HasLightGBMOptions> PMML encodePMML(E estimator){
		GBDT gbdt = getGBDT(estimator);

		Map<String, ?> options = getOptions(estimator);

		return gbdt.encodePMML(options, null, null);
	}

	static
	private GBDT getGBDT(HasBooster hasBooster){
		Booster booster = hasBooster.getBooster();

		return booster.getGBDT();
	}

	static
	private <E extends Estimator & HasBooster & HasLightGBMOptions> Map<String, ?> getOptions(E estimator){
		Map<String, Object> result = new LinkedHashMap<>();

		Boolean nanAsMissing = estimator.getPMMLOption(HasLightGBMOptions.OPTION_NAN_AS_MISSING, null);
		Integer numIteration = getNumIteration(estimator);

		result.put(HasLightGBMOptions.OPTION_NAN_AS_MISSING, nanAsMissing);
		result.put(HasLightGBMOptions.OPTION_NUM_ITERATION, numIteration);

		Boolean compact = estimator.getPMMLOption(HasLightGBMOptions.OPTION_COMPACT, null);
		Boolean flat = estimator.getPMMLOption(HasLightGBMOptions.OPTION_FLAT, null);
		Boolean nodeId = estimator.getPMMLOption(HasLightGBMOptions.OPTION_NODE_ID, null);
		Boolean nodeRecordCount = estimator.getPMMLOption(HasLightGBMOptions.OPTION_NODE_RECORDCOUNT, null);
		Boolean prune = estimator.getPMMLOption(HasLightGBMOptions.OPTION_PRUNE, null);

		result.put(HasLightGBMOptions.OPTION_COMPACT, compact);
		result.put(HasLightGBMOptions.OPTION_FLAT, flat);
		result.put(HasLightGBMOptions.OPTION_NODE_ID, nodeId);
		result.put(HasLightGBMOptions.OPTION_NODE_RECORDCOUNT, nodeRecordCount);
		result.put(HasLightGBMOptions.OPTION_PRUNE, prune);

		result.values().removeIf(Objects::isNull);

		return result;
	}

	static
	private <E extends Estimator & HasBooster & HasLightGBMOptions> Integer getNumIteration(E estimator){
		Integer bestIteration = estimator.getOptionalInteger("best_iteration_");

		return estimator.getPMMLOption(HasLightGBMOptions.OPTION_NUM_ITERATION, bestIteration);
	}
}