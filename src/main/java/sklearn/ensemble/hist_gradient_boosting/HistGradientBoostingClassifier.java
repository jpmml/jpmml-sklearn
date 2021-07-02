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
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import sklearn.Classifier;
import sklearn.Estimator;

public class HistGradientBoostingClassifier extends Classifier {

	public HistGradientBoostingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<? extends Number> baselinePredictions = getBaselinePrediction();
		BaseLoss loss = getLoss();
		BinMapper binMapper = getBinMapper();
		int numberOfTreesPerIteration = getNumberOfTreesPerIteration();
		List<List<TreePredictor>> predictors = getPredictors();

		if(predictors.size() > 0){
			ClassDictUtil.checkSize(numberOfTreesPerIteration, predictors.get(0), baselinePredictions);
		}

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(numberOfTreesPerIteration == 1){
			SchemaUtil.checkSize(2, categoricalLabel);

			if(!(loss instanceof BinaryCrossEntropy)){
				throw new IllegalArgumentException();
			}

			MiningModel miningModel = HistGradientBoostingUtil.encodeHistGradientBoosting(predictors, binMapper, baselinePredictions, 0, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(FieldNameUtil.create(Estimator.FIELD_DECISION_FUNCTION, categoricalLabel.getValue(1)), OpType.CONTINUOUS, DataType.DOUBLE));

			return MiningModelUtil.createBinaryLogisticClassification(miningModel, 1d, 0d, RegressionModel.NormalizationMethod.LOGIT, true, schema);
		} else

		if(numberOfTreesPerIteration >= 3){
			SchemaUtil.checkSize(numberOfTreesPerIteration, categoricalLabel);

			if(!(loss instanceof CategoricalCrossEntropy)){
				throw new IllegalArgumentException();
			}

			List<MiningModel> miningModels = new ArrayList<>();

			for(int i = 0, columns = categoricalLabel.size(); i < columns; i++){
				MiningModel miningModel = HistGradientBoostingUtil.encodeHistGradientBoosting(predictors, binMapper, baselinePredictions, i, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(FieldNameUtil.create(Estimator.FIELD_DECISION_FUNCTION, categoricalLabel.getValue(i)), OpType.CONTINUOUS, DataType.DOUBLE));

				miningModels.add(miningModel);
			}

			return MiningModelUtil.createClassification(miningModels, RegressionModel.NormalizationMethod.SOFTMAX, true, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<? extends Number> getBaselinePrediction(){
		return getNumberArray("_baseline_prediction");
	}

	public BinMapper getBinMapper(){
		return getOptional("_bin_mapper", BinMapper.class);
	}

	public BaseLoss getLoss(){

		// SkLearn 0.23
		if(containsKey("loss_")){
			get("loss_", BaseLoss.class);
		}

		// SkLearn 0.24+
		return get("_loss", BaseLoss.class);
	}

	public Integer getNumberOfTreesPerIteration(){
		return getInteger("n_trees_per_iteration_");
	}

	public List<List<TreePredictor>> getPredictors(){
		return (List)getList("_predictors", List.class);
	}
}