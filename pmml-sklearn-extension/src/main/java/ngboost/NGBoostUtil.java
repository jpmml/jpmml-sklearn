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
package ngboost;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;
import sklearn.EstimatorCastFunction;
import sklearn.Regressor;
import sklearn.tree.TreeRegressor;
import sklearn.tree.TreeUtil;

public class NGBoostUtil {

	private NGBoostUtil(){
	}

	static
	public <E extends Estimator> List<List<Regressor>> getBaseModels(E estimator){
		CastFunction<List> castFunction = new CastFunction<List>(List.class){

			private CastFunction<Regressor> castFunction = new EstimatorCastFunction<>(Regressor.class);


			@Override
			public List<Regressor> apply(Object object){
				List<?> values = super.apply(object);

				return Lists.transform(values, this.castFunction);
			}
		};

		return (List)estimator.getList("base_models", castFunction);
	}

	static
	public MiningModel encodeParamModel(int index, List<Number> initParams, List<List<Regressor>> baseModels, List<Number> scalings, Number learningRate, Schema schema){
		ContinuousLabel continuousLabel = schema.requireContinuousLabel();

		ClassDictUtil.checkSize(baseModels, scalings);

		PredicateManager predicateManager = new PredicateManager();

		List<Model> models = new ArrayList<>();

		if(!NGBoostUtil.isWeighted(scalings)){
			scalings = null;
		}

		for(int i = 0; i < baseModels.size(); i++){
			Regressor regressor = (baseModels.get(i)).get(index);

			Model model;

			if(regressor instanceof TreeRegressor){
				TreeRegressor treeRegressor = (TreeRegressor)regressor;

				model = TreeUtil.encodeTreeModel(treeRegressor, MiningFunction.REGRESSION, predicateManager, null, schema);
			} else

			{
				model = regressor.encodeModel(schema);
			}

			models.add(model);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation((scalings != null ? Segmentation.MultipleModelMethod.WEIGHTED_SUM : Segmentation.MultipleModelMethod.SUM), Segmentation.MissingPredictionTreatment.RETURN_MISSING, models, scalings))
			.setTargets(ModelUtil.createRescaleTargets(-learningRate.doubleValue(), initParams.get(index), continuousLabel));

		return miningModel;
	}

	static
	public ContinuousFeature encodePredictedParam(String name, Model model, PMMLEncoder encoder){
		OutputField outputField = ModelUtil.createPredictedField(name, OpType.CONTINUOUS, DataType.DOUBLE);

		DerivedOutputField derivedField = encoder.createDerivedField(model, outputField, true);

		return new ContinuousFeature(encoder, derivedField);
	}

	static
	public RegressionModel encodeRegression(Feature locFeature, RegressionModel.NormalizationMethod normalizationMethod, Schema schema){
		RegressionModel regressionModel = RegressionModelUtil.createRegression(Collections.singletonList(locFeature), Collections.singletonList(1d), null, normalizationMethod, schema);

		return regressionModel;
	}

	static
	public boolean isWeighted(List<Number> scalings){

		for(Number scaling : scalings){

			if(scaling.doubleValue() != 1d){
				return true;
			}
		}

		return false;
	}
}