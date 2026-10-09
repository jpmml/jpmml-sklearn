/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.dummy;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.primitives.Doubles;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.ClassifierNode;
import org.dmg.pmml.tree.ComplexNode;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.PayloadManager;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import sklearn.HasPriorProbability;
import sklearn.HasSparseOutput;
import sklearn.SkLearnClassifier;

public class DummyClassifier extends SkLearnClassifier implements HasPriorProbability, HasSparseOutput {

	public DummyClassifier(){
		this("sklearn.dummy", "DummyClassifier");
	}

	public DummyClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Number getPriorProbability(int index){
		List<?> classes = getClasses();
		List<Number> classPrior = getClassPrior();
		String strategy = getStrategy();

		ClassDictUtil.checkSize(classes, classPrior);

		switch(strategy){
			case DummyClassifier.STRATEGY_PRIOR:
				{
					return classPrior.get(index);
				}
			default:
				throw new IllegalArgumentException(strategy);
		}
	}

	@Override
	public TreeModel encodeModel(Schema schema){
		String strategy = getStrategy();

		PayloadManager payloadManager = new PayloadManager();

		int numberOfOutputs = getNumberOfOutputs();
		if(numberOfOutputs == 1){
			List<?> classes = getClasses();
			List<Number> classPrior = getClassPrior();
			Object constant = getConstant();

			CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();

			List<? extends Number> probabilities = encodeProbabilities(strategy, classes, classPrior, constant);

			int maxIndex = PayloadManager.indexOfMax((List)probabilities);

			Node root = new ClassifierNode(categoricalLabel.getValue(maxIndex), True.INSTANCE);

			payloadManager.addScoreDistributions(root, null, probabilities, categoricalLabel);

			TreeModel treeModel = new TreeModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), root);

			encodePredictProbaOutput(treeModel, DataType.DOUBLE, categoricalLabel);

			return treeModel;
		} else

		if(numberOfOutputs >= 2){
			List<List<?>> classes = (List)getClasses();
			List<List<Number>> classPrior = getArrayList("class_prior_", Number.class);
			List<?> constant = (DummyClassifier.STRATEGY_CONSTANT).equals(strategy) ? getListLike("constant") : Collections.nCopies(numberOfOutputs, null);

			ClassDictUtil.checkSize(numberOfOutputs, classes, classPrior, constant);

			MultiLabel multiLabel = schema.requireMultiLabel()
				.expectSize(numberOfOutputs);

			List<CategoricalLabel> categoricalLabels = multiLabel.toScalarLabels(CategoricalLabel.class);

			Node root = new ComplexNode(null, True.INSTANCE);

			List<ScoreDistribution> scoreDistributions = root.getScoreDistributions();

			for(int i = 0; i < numberOfOutputs; i++){
				CategoricalLabel categoricalLabel = categoricalLabels.get(i);

				List<?> targetClasses = classes.get(i);
				List<Number> targetClassPrior = classPrior.get(i);
				Object targetConstant = constant.get(i);

				String name = categoricalLabel.getName();

				List<? extends Number> probabilities = encodeProbabilities(strategy, targetClasses, targetClassPrior, targetConstant);

				for(int j = 0; j < categoricalLabel.size(); j++){
					Object value = categoricalLabel.getValue(j);
					Number probability = probabilities.get(j);

					ScoreDistribution scoreDistribution = payloadManager.createScoreProbability(name, value, null, probability);

					scoreDistributions.add(scoreDistribution);
				}
			}

			TreeModel treeModel = new TreeModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(multiLabel), root);

			encodePredictProbaOutput(treeModel, DataType.DOUBLE, multiLabel);

			return treeModel;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<Number> getClassPrior(){
		return getNumberArray("class_prior_");
	}

	public Object getConstant(){
		return getOptionalObject("constant");
	}

	@Override
	public Boolean getSparseOutput(){
		return getBoolean("sparse_output_");
	}

	public String getStrategy(){
		return getEnum("strategy", this::getString, Arrays.asList(DummyClassifier.STRATEGY_CONSTANT, DummyClassifier.STRATEGY_MOST_FREQUENT, DummyClassifier.STRATEGY_PRIOR));
	}

	static
	private List<? extends Number> encodeProbabilities(String strategy, List<?> classes, List<Number> classPrior, Object constant){
		ClassDictUtil.checkSize(classes, classPrior);

		switch(strategy){
			case DummyClassifier.STRATEGY_CONSTANT:
				{
					int index = classes.indexOf(constant);
					if(index < 0){
						throw new IllegalArgumentException();
					}

					return createProbabilities(classes, index);
				}
			case DummyClassifier.STRATEGY_MOST_FREQUENT:
				{
					int index = PayloadManager.indexOfMax((List)classPrior);

					return createProbabilities(classes, index);
				}
			case DummyClassifier.STRATEGY_PRIOR:
				{
					return classPrior;
				}
			default:
				throw new IllegalArgumentException(strategy);
		}
	}

	static
	private List<Double> createProbabilities(List<?> classes, int index){
		double[] values = new double[classes.size()];

		values[index] = 1d;

		return Doubles.asList(values);
	}

	private static final String STRATEGY_CONSTANT = "constant";
	private static final String STRATEGY_MOST_FREQUENT = "most_frequent";
	private static final String STRATEGY_PRIOR = "prior";
}