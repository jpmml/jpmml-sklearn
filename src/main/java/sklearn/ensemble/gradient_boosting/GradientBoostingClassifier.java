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

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.TreeModelProducer;
import sklearn.Classifier;
import sklearn.EstimatorUtil;
import sklearn.tree.DecisionTreeRegressor;

public class GradientBoostingClassifier extends Classifier implements TreeModelProducer {

	public GradientBoostingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){

		// SkLearn 0.18
		if(containsKey("n_features")){
			return ValueUtil.asInt((Number)get("n_features"));
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
		LossFunction loss = getLoss();

		int numberOfClasses = loss.getK();

		HasPriorProbability init = getInit();

		Number learningRate = getLearningRate();

		List<DecisionTreeRegressor> estimators = getEstimators();

		Schema segmentSchema = new Schema(new ContinuousLabel(null, DataType.DOUBLE), schema.getFeatures());

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(numberOfClasses == 1){
			EstimatorUtil.checkSize(2, categoricalLabel);

			MiningModel miningModel = GradientBoostingUtil.encodeGradientBoosting(estimators, init.getPriorProbability(0), learningRate, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction(" + categoricalLabel.getValue(1) + ")"), OpType.CONTINUOUS, DataType.DOUBLE, loss.createTransformation()));

			return MiningModelUtil.createBinaryLogisticClassification(miningModel, 1d, 0d, RegressionModel.NormalizationMethod.NONE, true, schema);
		} else

		if(numberOfClasses >= 3){
			EstimatorUtil.checkSize(numberOfClasses, categoricalLabel);

			List<MiningModel> miningModels = new ArrayList<>();

			for(int i = 0, columns = categoricalLabel.size(), rows = (estimators.size() / columns); i < columns; i++){
				MiningModel miningModel = GradientBoostingUtil.encodeGradientBoosting(CMatrixUtil.getColumn(estimators, rows, columns, i), init.getPriorProbability(i), learningRate, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction(" + categoricalLabel.getValue(i) + ")"), OpType.CONTINUOUS, DataType.DOUBLE, loss.createTransformation()));

				miningModels.add(miningModel);
			}

			return MiningModelUtil.createClassification(miningModels, RegressionModel.NormalizationMethod.SIMPLEMAX, true, schema);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public LossFunction getLoss(){
		Object loss = get("loss_");

		try {
			if(loss == null){
				throw new NullPointerException();
			}

			return (LossFunction)loss;
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The loss function object (" + ClassDictUtil.formatClass(loss) + ") is not a LossFunction or is not a supported LossFunction subclass", re);
		}
	}

	public HasPriorProbability getInit(){
		Object init = get("init_");

		try {
			if(init == null){
				throw new NullPointerException();
			}

			return (HasPriorProbability)init;
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(init) + ") is not a BaseEstimator or is not a supported BaseEstimator subclass", re);
		}
	}

	public Number getLearningRate(){
		return (Number)get("learning_rate");
	}

	public List<DecisionTreeRegressor> getEstimators(){
		return (List)ClassDictUtil.getArray(this, "estimators_");
	}
}