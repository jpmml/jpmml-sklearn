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
import java.util.function.IntFunction;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.AttributeException;
import org.jpmml.python.PythonObject;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;
import sklearn.HasMultiDecisionFunctionField;
import sklearn.HasPriorProbability;
import sklearn.SkLearnClassifier;
import sklearn.VersionUtil;
import sklearn.loss.BaseLoss;
import sklearn.loss.HalfLogitLink;
import sklearn.loss.Link;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.TreeRegressor;
import sklearn.tree.TreeUtil;
import sklearn2pmml.EstimatorProxy;

public class GradientBoostingClassifier extends SkLearnClassifier implements HasEstimatorEnsemble<TreeRegressor>, HasMultiDecisionFunctionField, HasTreeOptions {

	public GradientBoostingClassifier(String module, String name){
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
		String sklearnVersion = getSkLearnVersion();
		HasPriorProbability init = getInit();
		Number learningRate = getLearningRate();
		PythonObject loss = getLoss();

		IntFunction<Number> initialPredictions = init::getPriorProbability;

		if(loss instanceof LossFunction){
			LossFunction lossFunction = (LossFunction)loss;

			if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "0.21") >= 0){
				List<? extends Number> computedInitialPredictions = lossFunction.computeInitialPredictions(init);

				initialPredictions = computedInitialPredictions::get;
			}
		} else

		if(loss instanceof BaseLoss){
			BaseLoss baseLoss = (BaseLoss)loss;

			if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "1.4.0") >= 0){
				Link link = baseLoss.getLink();
				int numClasses = baseLoss.getNumClasses();

				List<? extends Number> computedInitialPredictions = link.computeInitialPredictions(numClasses, init);

				initialPredictions = computedInitialPredictions::get;
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		Transformation[] transformations = {};

		if(loss instanceof LossFunction){
			LossFunction lossFunction = (LossFunction)loss;

			transformations = new Transformation[]{lossFunction.createTransformation()};
		}

		MiningModel miningModel;

		if(categoricalLabel.size() == 2){
			SchemaUtil.checkSize(2, categoricalLabel);

			Model model = GradientBoostingUtil.encodeGradientBoosting(this, initialPredictions.apply(1), learningRate, segmentSchema)
				.setOutput(ModelUtil.createPredictedOutput(getMultiDecisionFunctionField(categoricalLabel.getValue(1)), OpType.CONTINUOUS, DataType.DOUBLE, transformations));

			double coefficient = 1d;

			RegressionModel.NormalizationMethod normalizationMethod = RegressionModel.NormalizationMethod.NONE;

			if(loss instanceof BaseLoss){
				BaseLoss baseLoss = (BaseLoss)loss;

				Link link = baseLoss.getLink();

				normalizationMethod = RegressionModel.NormalizationMethod.LOGIT;

				if(link instanceof HalfLogitLink){
					coefficient = 2d;
				}
			}

			miningModel = MiningModelUtil.createBinaryLogisticClassification(model, coefficient, 0d, normalizationMethod, false, schema);
		} else

		if(categoricalLabel.size() > 2){
			List<TreeRegressor> estimators = getEstimators();

			List<Model> models = new ArrayList<>();

			for(int i = 0, columns = categoricalLabel.size(), rows = (estimators.size() / columns); i < columns; i++){
				List<TreeRegressor> columnEstimators = CMatrixUtil.getColumn(estimators, rows, columns, i);

				GradientBoostingClassifierProxy estimatorProxy = new GradientBoostingClassifierProxy(){

					@Override
					public List<TreeRegressor> getEstimators(){
						return columnEstimators;
					}
				};

				Model model = GradientBoostingUtil.encodeGradientBoosting(estimatorProxy, initialPredictions.apply(i), learningRate, segmentSchema)
					.setOutput(ModelUtil.createPredictedOutput(getMultiDecisionFunctionField(categoricalLabel.getValue(i)), OpType.CONTINUOUS, DataType.DOUBLE, transformations));

				models.add(model);
			}

			RegressionModel.NormalizationMethod normalizationMethod = RegressionModel.NormalizationMethod.SIMPLEMAX;

			if(loss instanceof BaseLoss){
				normalizationMethod = RegressionModel.NormalizationMethod.SOFTMAX;
			}

			miningModel = MiningModelUtil.createClassification(models, normalizationMethod, false, schema);
		} else

		{
			throw new IllegalArgumentException();
		}

		encodePredictProbaOutput(miningModel, DataType.DOUBLE, categoricalLabel);

		return miningModel;
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

	public HasPriorProbability getInit(){
		return get("init_", HasPriorProbability.class);
	}

	public Number getLearningRate(){
		return getNumber("learning_rate");
	}

	public PythonObject getLoss(){

		// SkLearn 1.0.2
		if(hasattr("loss_")){
			return get("loss_", LossFunction.class);
		}

		// SkLearn 1.1.0+
		try {
			return get("_loss", LossFunction.class);

		// SkLearn 1.4.0+
		} catch(AttributeException ae){
			return get("_loss", sklearn.loss.BaseLoss.class);
		}
	}

	abstract
	private class GradientBoostingClassifierProxy extends EstimatorProxy implements HasEstimatorEnsemble<TreeRegressor>, HasTreeOptions {

		@Override
		public Estimator getEstimator(){
			return GradientBoostingClassifier.this;
		}
	}
}
