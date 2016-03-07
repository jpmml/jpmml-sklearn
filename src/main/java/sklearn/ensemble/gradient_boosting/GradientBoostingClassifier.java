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

import com.google.common.collect.Lists;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FeatureType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
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

		List<String> targetCategories = schema.getTargetCategories();

		Schema segmentSchema = EstimatorUtil.createSegmentSchema(schema);

		if(numberOfClasses == 1){

			if(targetCategories.size() != 2){
				throw new IllegalArgumentException();
			}

			targetCategories = Lists.reverse(targetCategories);

			double coefficient = loss.getCoefficient();

			MiningModel miningModel = encodeCategoryRegressor(targetCategories.get(0), estimators, init.getPriorProbability(0), learningRate, null, segmentSchema);

			return EstimatorUtil.encodeBinaryLogisticClassifier(targetCategories, miningModel, coefficient, true, schema);
		} else

		if(numberOfClasses >= 2){

			if(targetCategories.size() != numberOfClasses){
				throw new IllegalArgumentException();
			}

			List<MiningModel> miningModels = new ArrayList<>();

			for(int i = 0; i < targetCategories.size(); i++){
				MiningModel miningModel = encodeCategoryRegressor(targetCategories.get(i), NDArrayUtil.getColumn(estimators, estimators.size() / numberOfClasses, numberOfClasses, i), init.getPriorProbability(i), learningRate, loss.getFunction(), segmentSchema);

				miningModels.add(miningModel);
			}

			return EstimatorUtil.encodeMultinomialClassifier(targetCategories, miningModels, true, schema);
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
		OutputField decisionFunction = ModelUtil.createPredictedField(FieldName.create("decisionFunction_" + targetCategory));

		Output output = new Output()
			.addOutputFields(decisionFunction);

		if(outputTransformation != null){
			OutputField transformedDecisionField = new OutputField(FieldName.create(outputTransformation + "DecisionFunction_" + targetCategory))
				.setFeature(FeatureType.TRANSFORMED_VALUE)
				.setDataType(DataType.DOUBLE)
				.setOpType(OpType.CONTINUOUS)
				.setExpression(PMMLUtil.createApply(outputTransformation, new FieldRef(decisionFunction.getName())));

			output.addOutputFields(transformedDecisionField);
		}

		MiningModel miningModel = GradientBoostingUtil.encodeGradientBoosting(estimators, priorProbability, learningRate, schema)
			.setOutput(output);

		return miningModel;
	}
}