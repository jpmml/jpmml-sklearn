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

import java.util.List;
import java.util.function.Function;

import org.dmg.pmml.Model;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import sklearn.EstimatorUtil;
import sklearn.Regressor;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.TreeUtil;

public class BaseSRegressor extends BaseSLearner<Regressor> implements HasTreeOptions {

	public BaseSRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Class<Regressor> getEstimatorClass(){
		return Regressor.class;
	}

	@Override
	public Model encodeEstimator(Role role, Regressor regressor, Schema schema){
		Schema regressorSchema = toRegressorSchema(regressor, schema);

		return EstimatorUtil.encodeNativeLike(regressor, regressorSchema);
	}

	@Override
	public Schema configureSchema(Schema schema){
		Feature controlFeature = schema.getFeature(0);

		Function<Feature, Feature> function = Function.identity();

		Schema treeSchema = schema.toTransformedSchema(function);

		treeSchema = TreeUtil.configureSchema(this, treeSchema);

		// XXX
		List<Feature> treeFeatures = (List<Feature>)treeSchema.getFeatures();
		treeFeatures.set(0, controlFeature);

		return treeSchema;
	}

	@Override
	public Model configureModel(Model model){
		return TreeUtil.configureModel(this, model);
	}
}