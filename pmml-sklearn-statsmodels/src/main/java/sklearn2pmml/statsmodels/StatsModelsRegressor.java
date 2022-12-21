/*
 * Copyright (c) 2022 Villu Ruusmann
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
package sklearn2pmml.statsmodels;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.python.PickleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.statsmodels.InterceptFeature;
import sklearn.Regressor;
import statsmodels.regression.RegressionModel;
import statsmodels.regression.RegressionResults;
import statsmodels.regression.RegressionResultsWrapper;

public class StatsModelsRegressor extends Regressor {

	public StatsModelsRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		Boolean fitIntercept = getFitIntercept();
		RegressionModel model = getModel();
		RegressionResultsWrapper resultsWrapper = getResults();

		RegressionResults results = resultsWrapper.getResults();

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(fitIntercept){
			List<Feature> sklearnFeatures = new ArrayList<>(1 + features.size());
			sklearnFeatures.add(new InterceptFeature(encoder, "const", DataType.DOUBLE));
			sklearnFeatures.addAll(features);

			schema = new Schema(encoder, label, sklearnFeatures);
		}

		return results.encodeModel(schema);
	}

	public Boolean getFitIntercept(){
		return getBoolean("fit_intercept");
	}

	public RegressionModel getModel(){
		return get("model_", RegressionModel.class);
	}

	public RegressionResultsWrapper getResults(){
		return get("results_", RegressionResultsWrapper.class);
	}

	static {
		ClassLoader clazzLoader = SkLearnEncoder.class.getClassLoader();

		PickleUtil.init(clazzLoader, "statsmodels2pmml.properties");
	}
}