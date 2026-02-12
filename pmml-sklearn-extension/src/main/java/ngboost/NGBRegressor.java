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
import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Lists;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import sklearn.EstimatorCastFunction;
import sklearn.Regressor;

public class NGBRegressor extends Regressor {

	public NGBRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		@SuppressWarnings("unused")
		String distName = getDistName();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		ClassDictUtil.checkSize(baseModels, scalings);

		if(!isWeighted(scalings)){
			scalings = null;
		}

		Schema segmentSchema = schema.toAnonymousSchema();

		ContinuousLabel regressionLabel = new ContinuousLabel(null, DataType.DOUBLE);

		MiningModel locModel = encodeParamModel(0, baseModels, scalings, segmentSchema)
			.setTargets(ModelUtil.createRescaleTargets(-learningRate.doubleValue(), initParams.get(0), regressionLabel))
			.setOutput(ModelUtil.createPredictedOutput("loc", OpType.CONTINUOUS, DataType.DOUBLE));

		return MiningModelUtil.createRegression(locModel, RegressionModel.NormalizationMethod.NONE, schema);
	}

	public List<List<Regressor>> getBaseModels(){
		CastFunction<List> castFunction = new CastFunction<List>(List.class){

			@Override
			public List<Regressor> apply(Object object){
				List<?> values = super.apply(object);

				CastFunction<Regressor> castFunction = new EstimatorCastFunction<>(Regressor.class);

				return Lists.transform(values, castFunction);
			}
		};

		return (List)getList("base_models", castFunction);
	}

	public String getDistName(){
		return getEnum("Dist_name", this::getString, Arrays.asList(NGBRegressor.DIST_NORMAL));
	}

	public List<Number> getInitParams(){
		return getNumberListLike("init_params");
	}

	public Number getLearningRate(){
		return getNumber("learning_rate");
	}

	public List<Number> getScalings(){
		return getNumberListLike("scalings");
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

	static
	private MiningModel encodeParamModel(int index, List<List<Regressor>> baseModels, List<Number> scalings, Schema schema){
		List<Model> models = new ArrayList<>();

		for(int i = 0; i < baseModels.size(); i++){
			Regressor regressor = (baseModels.get(i)).get(index);

			Model model = regressor.encodeModel(schema);

			models.add(model);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation((scalings != null ? Segmentation.MultipleModelMethod.WEIGHTED_SUM : Segmentation.MultipleModelMethod.SUM), Segmentation.MissingPredictionTreatment.RETURN_MISSING, models, scalings));

		return miningModel;
	}

	private static final String DIST_NORMAL = "Normal";
}