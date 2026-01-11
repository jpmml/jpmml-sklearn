/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.ensemble.hist_gradient_boosting;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.AttributeException;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonObject;
import sklearn.HasMultiDecisionFunctionField;
import sklearn.SkLearnClassifier;
import sklearn.compose.ColumnTransformer;

public class HistGradientBoostingClassifier extends SkLearnClassifier implements HasMultiDecisionFunctionField {

	public HistGradientBoostingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<Number> baselinePredictions = getBaselinePrediction();
		@SuppressWarnings("unused")
		PythonObject loss = getLoss();
		BinMapper binMapper = getBinMapper();
		int numberOfTreesPerIteration = getNumberOfTreesPerIteration();
		List<List<TreePredictor>> predictors = getPredictors();
		ColumnTransformer preprocessor = getPreprocessor();

		if(!predictors.isEmpty()){
			ClassDictUtil.checkSize(numberOfTreesPerIteration, predictors.get(0), baselinePredictions);
		} // End if

		if(preprocessor != null){
			schema = HistGradientBoostingUtil.preprocess(preprocessor, schema);
		}

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();

		MiningModel miningModel;

		if(numberOfTreesPerIteration == 1){
			categoricalLabel.expectCardinality(2);

			Model model = HistGradientBoostingUtil.encodeHistGradientBoosting(predictors, binMapper, baselinePredictions, 0, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(getMultiDecisionFunctionField(categoricalLabel.getValue(1)), OpType.CONTINUOUS, DataType.DOUBLE));

			miningModel = MiningModelUtil.createBinaryLogisticClassification(model, 1d, 0d, RegressionModel.NormalizationMethod.LOGIT, false, schema);
		} else

		if(numberOfTreesPerIteration >= 3){
			categoricalLabel.expectCardinality(numberOfTreesPerIteration);

			List<Model> models = new ArrayList<>();

			for(int i = 0, columns = categoricalLabel.size(); i < columns; i++){
				Model model = HistGradientBoostingUtil.encodeHistGradientBoosting(predictors, binMapper, baselinePredictions, i, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(getMultiDecisionFunctionField(categoricalLabel.getValue(i)), OpType.CONTINUOUS, DataType.DOUBLE));

				models.add(model);
			}

			miningModel = MiningModelUtil.createClassification(models, RegressionModel.NormalizationMethod.SOFTMAX, false, schema);
		} else

		{
			throw new IllegalArgumentException();
		}

		encodePredictProbaOutput(miningModel, DataType.DOUBLE, categoricalLabel);

		return miningModel;
	}

	public List<Number> getBaselinePrediction(){
		return getNumberArray("_baseline_prediction");
	}

	public BinMapper getBinMapper(){
		return getOptional("_bin_mapper", BinMapper.class);
	}

	public PythonObject getLoss(){

		// SkLearn 0.23
		if(hasattr("loss_")){
			get("loss_", BaseLoss.class);
		}

		// SkLearn 0.24+
		try {
			return get("_loss", BaseLoss.class);

		// SkLearn 1.1.0+
		} catch(AttributeException ae){
			return get("_loss", sklearn.loss.BaseLoss.class);
		}
	}

	public Integer getNumberOfTreesPerIteration(){
		return getInteger("n_trees_per_iteration_");
	}

	public List<List<TreePredictor>> getPredictors(){
		return (List)getList("_predictors", List.class);
	}

	public ColumnTransformer getPreprocessor(){
		return getOptional("_preprocessor", ColumnTransformer.class);
	}
}