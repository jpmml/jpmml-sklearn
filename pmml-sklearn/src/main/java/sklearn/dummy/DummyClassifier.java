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

import java.util.List;

import com.google.common.primitives.Doubles;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.ClassifierNode;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ScoreDistributionManager;
import org.jpmml.python.ClassDictUtil;
import sklearn.Classifier;
import sklearn.HasPriorProbability;

public class DummyClassifier extends Classifier implements HasPriorProbability {

	public DummyClassifier(){
		this("sklearn.dummy", "DummyClassifier");
	}

	public DummyClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Number getPriorProbability(int index){
		List<?> classes = getClasses();
		List<? extends Number> classPrior = getClassPrior();
		String strategy = getStrategy();

		ClassDictUtil.checkSize(classes, classPrior);

		switch(strategy){
			case "prior":
				{
					return classPrior.get(index);
				}
			default:
				throw new IllegalArgumentException(strategy);
		}
	}

	@Override
	public TreeModel encodeModel(Schema schema){
		List<?> classes = getClasses();
		List<? extends Number> classPrior = getClassPrior();
		Object constant = getConstant();
		String strategy = getStrategy();

		ClassDictUtil.checkSize(classes, classPrior);

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		int maxIndex;

		List<? extends Number> probabilities;

		switch(strategy){
			case "constant":
				{
					maxIndex = classes.indexOf(constant);
					if(maxIndex < 0){
						throw new IllegalArgumentException();
					}

					probabilities = createProbabilities(classes, maxIndex);
				}
				break;
			case "most_frequent":
				{
					maxIndex = ScoreDistributionManager.indexOfMax((List)classPrior);

					probabilities = createProbabilities(classes, maxIndex);
				}
				break;
			case "prior":
				{
					maxIndex = ScoreDistributionManager.indexOfMax((List)classPrior);

					probabilities = classPrior;
				}
				break;
			default:
				throw new IllegalArgumentException(strategy);
		}

		Node root = new ClassifierNode(categoricalLabel.getValue(maxIndex), True.INSTANCE);

		ScoreDistributionManager scoreDistributionManager = new ScoreDistributionManager();

		scoreDistributionManager.addScoreDistributions(root, categoricalLabel.getValues(), null, probabilities);

		TreeModel treeModel = new TreeModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), root);

		encodePredictProbaOutput(treeModel, DataType.DOUBLE, categoricalLabel);

		return treeModel;
	}

	public List<? extends Number> getClassPrior(){
		return getNumberArray("class_prior_");
	}

	public Object getConstant(){
		return getOptionalScalar("constant");
	}

	public String getStrategy(){
		return getString("strategy");
	}

	static
	private List<Double> createProbabilities(List<?> classes, int index){
		double[] values = new double[classes.size()];

		values[index] = 1d;

		return Doubles.asList(values);
	}
}