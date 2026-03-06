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

import java.util.Collections;

import org.dmg.pmml.Model;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.EstimatorUtil;
import sklearn.Regressor;

public class BaseSRegressor extends BaseSLearner<Regressor> {

	public BaseSRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Class<Regressor> getEstimatorClass(){
		return Regressor.class;
	}

	@Override
	public Model encodeEstimator(Regressor regressor, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		ContinuousLabel continuousLabel = (ContinuousLabel)regressor.encodeLabel(Collections.singletonList(null), encoder);

		Schema regressorSchema = schema.toRelabeledSchema(continuousLabel);

		return EstimatorUtil.encodeNativeLike(regressor, regressorSchema);
	}
}