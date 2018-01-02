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
import sklearn.Classifier;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;
import sklearn.tree.DecisionTreeRegressor;
import sklearn.tree.HasTreeOptions;
import sklearn2pmml.EstimatorProxy;

public class GradientBoostingClassifier extends Classifier implements HasEstimatorEnsemble<DecisionTreeRegressor>, HasTreeOptions {

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

		Schema segmentSchema = new Schema(new ContinuousLabel(null, DataType.DOUBLE), schema.getFeatures());

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(numberOfClasses == 1){
			ClassifierUtil.checkSize(2, categoricalLabel);

			MiningModel miningModel = GradientBoostingUtil.encodeGradientBoosting(this, init.getPriorProbability(0), learningRate, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(FieldName.create("decisionFunction(" + categoricalLabel.getValue(1) + ")"), OpType.CONTINUOUS, DataType.DOUBLE, loss.createTransformation()));

			return MiningModelUtil.createBinaryLogisticClassification(miningModel, 1d, 0d, RegressionModel.NormalizationMethod.NONE, true, schema);
		} else

		if(numberOfClasses >= 3){
			ClassifierUtil.checkSize(numberOfClasses, categoricalLabel);

			List<? extends DecisionTreeRegressor> estimators = getEstimators();

			List<MiningModel> miningModels = new ArrayList<>();

			for(int i = 0, columns = categoricalLabel.size(), rows = (estimators.size() / columns); i < columns; i++){
				final
				List<? extends DecisionTreeRegressor> columnEstimators = CMatrixUtil.getColumn(estimators, rows, columns, i);

				GradientBoostingClassifierProxy estimatorProxy = new GradientBoostingClassifierProxy(){

					@Override
					public List<? extends DecisionTreeRegressor> getEstimators(){
						return columnEstimators;
					}
				};

				MiningModel miningModel = GradientBoostingUtil.encodeGradientBoosting(estimatorProxy, init.getPriorProbability(i), learningRate, segmentSchema)
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
		return get("loss_", LossFunction.class);
	}

	public HasPriorProbability getInit(){
		return get("init_", HasPriorProbability.class);
	}

	public Number getLearningRate(){
		return (Number)get("learning_rate");
	}

	@Override
	public List<? extends DecisionTreeRegressor> getEstimators(){
		return getArray("estimators_", DecisionTreeRegressor.class);
	}

	abstract
	private class GradientBoostingClassifierProxy extends EstimatorProxy implements HasEstimatorEnsemble<DecisionTreeRegressor>, HasTreeOptions {

		@Override
		public Estimator getEstimator(){
			return GradientBoostingClassifier.this;
		}
	}
}