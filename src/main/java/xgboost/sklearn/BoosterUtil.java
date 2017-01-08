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

import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public int getNumberOfFeatures(HasBooster hasBooster){
		Booster booster = hasBooster.getBooster();

		Learner learner = booster.getLearner();

		return learner.getNumFeatures();
	}

	static
	public MiningModel encodeBooster(HasBooster hasBooster, Schema schema){
		Booster booster = hasBooster.getBooster();

		Learner learner = booster.getLearner();

		Schema xgbSchema = XGBoostUtil.toXGBoostSchema(schema);

		MiningModel miningModel = learner.encodeMiningModel(xgbSchema);

		return miningModel;
	}
}