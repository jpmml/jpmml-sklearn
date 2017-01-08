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
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.MatrixUtil;
import sklearn.Classifier;
import sklearn.EstimatorUtil;
import sklearn.tree.DecisionTreeRegressor;

public class GradientBoostingClassifier extends Classifier {

	public GradientBoostingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return ValueUtil.asInt((Number)get("n_features"));
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

		Schema segmentSchema = schema.toAnonymousSchema();

		CategoricalLabel categoricalLabel = (CategoricalLabel)segmentSchema.getLabel();

		if(numberOfClasses == 1){
			EstimatorUtil.checkSize(2, categoricalLabel);

			double coefficient = loss.getCoefficient();

			MiningModel miningModel = encodeCategoryRegressor(categoricalLabel.getValue(1), estimators, init.getPriorProbability(0), learningRate, null, segmentSchema);

			return MiningModelUtil.createBinaryLogisticClassification(schema, miningModel, coefficient, true);
		} else

		if(numberOfClasses >= 2){
			EstimatorUtil.checkSize(numberOfClasses, categoricalLabel);

			List<MiningModel> miningModels = new ArrayList<>();

			for(int i = 0, columns = categoricalLabel.size(), rows = (estimators.size() / columns); i < columns; i++){
				MiningModel miningModel = encodeCategoryRegressor(categoricalLabel.getValue(i), MatrixUtil.getColumn(estimators, rows, columns, i), init.getPriorProbability(i), learningRate, loss.getFunction(), segmentSchema);

				miningModels.add(miningModel);
			}

			return MiningModelUtil.createClassification(schema, miningModels, RegressionModel.NormalizationMethod.SIMPLEMAX, true);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	@Override
	public Set<DefineFunction> encodeDefineFunctions(){
		LossFunction loss = getLoss();

		DefineFunction defineFunction = loss.encodeFunction();
		if(defineFunction != null){
			return Collections.singleton(defineFunction);
		}

		return super.encodeDefineFunctions();
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

	static
	private MiningModel encodeCategoryRegressor(String targetCategory, List<DecisionTreeRegressor> estimators, Number priorProbability, Number learningRate, String outputTransformation, Schema schema){
		OutputField decisionFunction = new OutputField(FieldName.create("decisionFunction_" + targetCategory), DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setResultFeature(ResultFeature.PREDICTED_VALUE)
			.setFinalResult(false);

		Output output = new Output()
			.addOutputFields(decisionFunction);

		if(outputTransformation != null){
			OutputField transformedDecisionField = new OutputField(FieldName.create(outputTransformation + "DecisionFunction_" + targetCategory), DataType.DOUBLE)
				.setOpType(OpType.CONTINUOUS)
				.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
				.setFinalResult(false)
				.setExpression(PMMLUtil.createApply(outputTransformation, new FieldRef(decisionFunction.getName())));

			output.addOutputFields(transformedDecisionField);
		}

		MiningModel miningModel = GradientBoostingUtil.encodeGradientBoosting(estimators, priorProbability, learningRate, schema)
			.setOutput(output);

		return miningModel;
	}
}