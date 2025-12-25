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

import org.dmg.pmml.PMML;
import org.dmg.pmml.mining.MiningModel;
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
	public <E extends Estimator & HasBooster & HasLightGBMOptions> MiningModel encodeModel(E estimator, Schema schema){
		GBDT gbdt = getGBDT(estimator);

		ModelEncoder encoder = schema.getEncoder();

		Map<String, ?> options = getOptions(gbdt, estimator);

		Integer numIterations = (Integer)options.get(HasLightGBMOptions.OPTION_NUM_ITERATION);

		MiningModel miningModel = gbdt.encodeModel(numIterations, schema);

		encoder.transferFeatureImportances(null, miningModel);

		return miningModel;
	}

	static
	public <E extends Estimator & HasBooster & HasLightGBMOptions> Schema configureSchema(E estimator, Schema schema){
		GBDT gbdt = getGBDT(estimator);

		Map<String, ?> options = getOptions(gbdt, estimator);

		Schema lgbmSchema = gbdt.toLightGBMSchema(schema);

		lgbmSchema = gbdt.configureSchema(options, lgbmSchema);

		return lgbmSchema;
	}

	static
	public <E extends Estimator & HasBooster & HasLightGBMOptions> MiningModel configureModel(E estimator, MiningModel miningModel){
		GBDT gbdt = getGBDT(estimator);

		Map<String, ?> options = getOptions(gbdt, estimator);

		miningModel = gbdt.configureModel(options, miningModel);

		return miningModel;
	}

	static
	public PMML encodePMML(Booster booster){
		GBDT gbdt = booster.getGBDT();

		Map<String, ?> options = getOptions(gbdt);

		return gbdt.encodePMML(options, null, null);
	}

	static
	public <E extends Estimator & HasBooster & HasLightGBMOptions> PMML encodePMML(E estimator){
		GBDT gbdt = getGBDT(estimator);

		Map<String, ?> options = getOptions(gbdt, estimator);

		return gbdt.encodePMML(options, null, null);
	}

	static
	private GBDT getGBDT(HasBooster hasBooster){
		Booster booster = hasBooster.getBooster();

		return booster.getGBDT();
	}

	static
	private Map<String, ?> getOptions(GBDT gbdt){
		Map<String, Object> result = new LinkedHashMap<>();

		Boolean nanAsMissing = Boolean.TRUE;
		result.put(HasLightGBMOptions.OPTION_NAN_AS_MISSING, nanAsMissing);

		return result;
	}

	static
	private <E extends Estimator & HasBooster & HasLightGBMOptions> Map<String, ?> getOptions(GBDT gbdt, E estimator){
		Map<String, Object> result = new LinkedHashMap<>();

		Integer bestIteration = estimator.getOptionalInteger("best_iteration_");

		Integer numIteration = (Integer)estimator.getOption(HasLightGBMOptions.OPTION_NUM_ITERATION, bestIteration);
		result.put(HasLightGBMOptions.OPTION_NUM_ITERATION, numIteration);

		Boolean nanAsMissing = (Boolean)estimator.getOption(HasLightGBMOptions.OPTION_NAN_AS_MISSING, Boolean.TRUE);
		result.put(HasLightGBMOptions.OPTION_NAN_AS_MISSING, nanAsMissing);

		Boolean compact = (Boolean)estimator.getOption(HasLightGBMOptions.OPTION_COMPACT, Boolean.TRUE);
		result.put(HasLightGBMOptions.OPTION_COMPACT, compact);

		return result;
	}
}