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

import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.lightgbm.GBDT;
import org.jpmml.lightgbm.HasLightGBMOptions;
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
	public <E extends Estimator & HasBooster & HasLightGBMOptions> MiningModel encodeModel(E estimator, Schema schema){
		GBDT gbdt = getGBDT(estimator);

		Integer bestIteration = (Integer)estimator.getOptionalScalar("best_iteration_");

		Boolean compact = (Boolean)estimator.getOption(HasLightGBMOptions.OPTION_COMPACT, Boolean.TRUE);
		Integer numIteration = (Integer)estimator.getOption(HasLightGBMOptions.OPTION_NUM_ITERATION, bestIteration);

		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasLightGBMOptions.OPTION_COMPACT, compact);
		options.put(HasLightGBMOptions.OPTION_NUM_ITERATION, numIteration);

		Schema lgbmSchema = gbdt.toLightGBMSchema(schema);

		MiningModel miningModel = gbdt.encodeMiningModel(options, lgbmSchema);

		return miningModel;
	}

	static
	private GBDT getGBDT(HasBooster hasBooster){
		Booster booster = hasBooster.getBooster();

		return booster.getGBDT();
	}
}