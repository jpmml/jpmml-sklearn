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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
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
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import org.jpmml.sklearn.SchemaUtil;
import sklearn.Classifier;
import sklearn.EstimatorUtil;
import sklearn.tree.DecisionTreeRegressor;

public class GradientBoostingClassifier extends Classifier {

	public GradientBoostingClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		return (Integer)get("n_features");
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

		Function<MiningModel, FieldName> probabilityFieldFunction = new Function<MiningModel, FieldName>(){

			@Override
			public FieldName apply(MiningModel miningModel){
				Output output = miningModel.getOutput();

				OutputField outputField = Iterables.getLast(output.getOutputFields());

				return outputField.getName();
			}
		};

		Schema segmentSchema = SchemaUtil.createSegmentSchema(schema);

		if(numberOfClasses == 1){

			if(targetCategories.size() != 2){
				throw new IllegalArgumentException();
			}

			targetCategories = Lists.reverse(targetCategories);

			MiningModel miningModel = encodeCategoryRegressor(targetCategories.get(0), loss, estimators, init.getPriorProbability(0), learningRate, segmentSchema);

			List<FieldName> probabilityFields = new ArrayList<>();
			probabilityFields.add(probabilityFieldFunction.apply(miningModel));
			probabilityFields.add(FieldName.create(loss.getFunction() + "DecisionFunction_" + targetCategories.get(1)));

			return EstimatorUtil.encodeBinomialClassifier(targetCategories, probabilityFields, miningModel, true, schema);
		} else

		if(numberOfClasses >= 2){

			if(targetCategories.size() != numberOfClasses){
				throw new IllegalArgumentException();
			}

			List<MiningModel> miningModels = new ArrayList<>();

			for(int i = 0; i < targetCategories.size(); i++){
				MiningModel miningModel = encodeCategoryRegressor(targetCategories.get(i), loss, NDArrayUtil.getColumn(estimators, estimators.size() / numberOfClasses, numberOfClasses, i), init.getPriorProbability(i), learningRate, segmentSchema);

				miningModels.add(miningModel);
			}

			List<FieldName> probabilityFields = Lists.transform(miningModels, probabilityFieldFunction);

			return EstimatorUtil.encodeMultinomialClassifier(targetCategories, probabilityFields, miningModels, true, schema);
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
	private MiningModel encodeCategoryRegressor(String targetCategory, LossFunction loss, List<DecisionTreeRegressor> estimators, Number priorProbability, Number learningRate, Schema schema){
		OutputField decisionFunction = PMMLUtil.createPredictedField(FieldName.create("decisionFunction_" + targetCategory));

		OutputField transformedDecisionField = new OutputField(FieldName.create(loss.getFunction() + "DecisionFunction_" + targetCategory))
			.setFeature(FeatureType.TRANSFORMED_VALUE)
			.setDataType(DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setExpression(PMMLUtil.createApply(loss.getFunction(), new FieldRef(decisionFunction.getName())));

		Output output = new Output()
			.addOutputFields(decisionFunction, transformedDecisionField);

		MiningModel miningModel = GradientBoostingUtil.encodeGradientBoosting(estimators, priorProbability, learningRate, schema)
			.setOutput(output);

		return miningModel;
	}
}