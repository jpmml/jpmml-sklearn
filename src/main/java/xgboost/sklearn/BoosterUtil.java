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
package xgboost.sklearn;

import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;

import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.xgboost.ByteOrderUtil;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.jpmml.xgboost.Learner;
import sklearn.Estimator;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> int getNumberOfFeatures(E estimator){
		Learner learner = getLearner(estimator);

		return learner.num_features();
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> MiningModel encodeBooster(E estimator, Schema schema){
		Learner learner = getLearner(estimator);

		Integer bestNTreeLimit = (Integer)estimator.getOptionalScalar("best_ntree_limit");

		Boolean compact = (Boolean)estimator.getOption(HasXGBoostOptions.OPTION_COMPACT, Boolean.TRUE);
		Integer ntreeLimit = (Integer)estimator.getOption(HasXGBoostOptions.OPTION_NTREE_LIMIT, bestNTreeLimit);

		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasXGBoostOptions.OPTION_COMPACT, compact);
		options.put(HasXGBoostOptions.OPTION_NTREE_LIMIT, ntreeLimit);

		Schema xgbSchema = learner.toXGBoostSchema(schema);

		MiningModel miningModel = learner.encodeMiningModel(options, xgbSchema);

		return miningModel;
	}

	static
	private <E extends Estimator & HasBooster & HasXGBoostOptions> Learner getLearner(E estimator){
		Booster booster = estimator.getBooster();

		String byteOrder = (String)estimator.getOption(HasXGBoostOptions.OPTION_BYTE_ORDER, (ByteOrder.nativeOrder()).toString());
		String charset = (String)estimator.getOption(HasXGBoostOptions.OPTION_CHARSET, null);

		return booster.getLearner(ByteOrderUtil.forValue(byteOrder), charset);
	}
}