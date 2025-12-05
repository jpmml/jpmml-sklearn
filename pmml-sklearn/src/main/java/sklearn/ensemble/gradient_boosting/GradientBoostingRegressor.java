/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.ensemble.gradient_boosting;

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import sklearn.HasDefaultValue;
import sklearn.HasEstimatorEnsemble;
import sklearn.SkLearnRegressor;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.TreeRegressor;
import sklearn.tree.TreeUtil;

public class GradientBoostingRegressor extends SkLearnRegressor implements HasEstimatorEnsemble<TreeRegressor>, HasTreeOptions {

	public GradientBoostingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){

		// SkLearn 0.18
		if(hasattr("n_features")){
			return getInteger("n_features");
		}

		// SkLearn 0.19+
		return super.getNumberOfFeatures();
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		HasDefaultValue init = getInit();
		Number learningRate = getLearningRate();

		return GradientBoostingUtil.encodeGradientBoosting(this, init.getDefaultValue(), learningRate, schema);
	}

	@Override
	public Schema configureSchema(Schema schema){
		return TreeUtil.configureSchema(this, schema);
	}

	@Override
	public Model configureModel(Model model){
		return TreeUtil.configureModel(this, model);
	}

	@Override
	public List<TreeRegressor> getEstimators(){
		return getEstimatorArray("estimators_", TreeRegressor.class);
	}

	public HasDefaultValue getInit(){
		return get("init_", HasDefaultValue.class);
	}

	public Number getLearningRate(){
		return getNumber("learning_rate");
	}
}