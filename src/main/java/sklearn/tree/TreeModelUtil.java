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
package sklearn.tree;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import numpy.core.ScalarUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.visitors.AbstractExtender;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.visitors.TreeModelCompactor;
import org.jpmml.sklearn.visitors.TreeModelFlattener;
import sklearn.Estimator;
import sklearn.HasEstimatorEnsemble;

public class TreeModelUtil {

	private TreeModelUtil(){
	}

	static
	public <E extends Estimator & HasTreeOptions, M extends Model> M transform(E estimator, M model){
		Boolean winnerId = (Boolean)estimator.getOption(HasTreeOptions.OPTION_WINNER_ID, Boolean.FALSE);

		Map<String, Map<Integer, ?>> nodeExtensions = (Map)estimator.getOption(HasTreeOptions.OPTION_NODE_EXTENSIONS, null);
		Boolean nodeId = (Boolean)estimator.getOption(HasTreeOptions.OPTION_NODE_ID, winnerId);

		boolean fixed = (nodeExtensions != null || nodeId);

		Boolean compact = (Boolean)estimator.getOption(HasTreeOptions.OPTION_COMPACT, fixed ? Boolean.FALSE : Boolean.TRUE);
		Boolean flat = (Boolean)estimator.getOption(HasTreeOptions.OPTION_FLAT, Boolean.FALSE);

		if(fixed && (compact || flat)){
			throw new IllegalArgumentException("Conflicting tree model options");
		} // End if

		if((Boolean.TRUE).equals(winnerId)){
			Output output = ModelUtil.ensureOutput(model);

			OutputField nodeIdField = ModelUtil.createEntityIdField(FieldName.create("nodeId"))
				.setDataType(DataType.INTEGER);

			output.addOutputFields(nodeIdField);
		}

		List<Visitor> visitors = new ArrayList<>();

		if((Boolean.TRUE).equals(compact)){
			visitors.add(new TreeModelCompactor());
		} // End if

		if((Boolean.TRUE).equals(flat)){
			visitors.add(new TreeModelFlattener());
		} // End if

		if(nodeExtensions != null){
			Collection<? extends Map.Entry<String, Map<Integer, ?>>> entries = nodeExtensions.entrySet();

			for(Map.Entry<String, Map<Integer, ?>> entry : entries){
				String name = entry.getKey();

				Map<Integer, ?> values = entry.getValue();

				Visitor nodeExtender = new AbstractExtender(name){

					@Override
					public VisitorAction visit(Node node){
						Integer id = Integer.valueOf(node.getId());

						Object value = values.get(id);
						if(value != null){
							value = ScalarUtil.decode(value);

							addExtension(node, ValueUtil.formatValue(value));
						}

						return super.visit(node);
					}
				};

				visitors.add(nodeExtender);
			}
		} // End if

		if((Boolean.FALSE).equals(nodeId)){
			Visitor nodeIdCleaner = new AbstractVisitor(){

				@Override
				public VisitorAction visit(Node node){
					node.setId(null);

					return super.visit(node);
				}
			};

			visitors.add(nodeIdCleaner);
		}

		for(Visitor visitor : visitors){
			visitor.applyTo(model);
		}

		return model;
	}

	static
	public <E extends Estimator & HasEstimatorEnsemble<T>, T extends Estimator & HasTree> List<TreeModel> encodeTreeModelSegmentation(E estimator, MiningFunction miningFunction, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeTreeModelSegmentation(estimator, predicateManager, miningFunction, schema);
	}

	static
	public <E extends Estimator & HasEstimatorEnsemble<T>, T extends Estimator & HasTree> List<TreeModel> encodeTreeModelSegmentation(E estimator, final PredicateManager predicateManager, final MiningFunction miningFunction, final Schema schema){
		List<? extends T> estimators = estimator.getEstimators();

		final
		Schema segmentSchema = schema.toAnonymousSchema();

		Function<T, TreeModel> function = new Function<T, TreeModel>(){

			@Override
			public TreeModel apply(T estimator){
				Schema treeModelSchema = toTreeModelSchema(estimator.getDataType(), segmentSchema);
				TreeModel model = TreeModelUtil.encodeTreeModel(estimator, predicateManager, miningFunction, treeModelSchema);
				estimator.getTree().freeResources();
				return model;
			}
		};

		return estimators.stream()
			.map(function)
			.collect(Collectors.toList());
	}

	static
	public <E extends Estimator & HasTree> TreeModel encodeTreeModel(E estimator, MiningFunction miningFunction, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeTreeModel(estimator, predicateManager, miningFunction, schema);
	}

	static
	public <E extends Estimator & HasTree> TreeModel encodeTreeModel(E estimator, PredicateManager predicateManager, MiningFunction miningFunction, Schema schema){
		Tree tree = estimator.getTree();

		int[] leftChildren = tree.getChildrenLeft();
		int[] rightChildren = tree.getChildrenRight();
		int[] features = tree.getFeature();
		double[] thresholds = tree.getThreshold();
		double[] values = tree.getValues();

		Node root = new Node()
			.setPredicate(new True());

		encodeNode(root, predicateManager, 0, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

		TreeModel treeModel = new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema.getLabel()), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT);

		return treeModel;
	}

	static
	private void encodeNode(Node node, PredicateManager predicateManager, int index, int[] leftChildren, int[] rightChildren, int[] features, double[] thresholds, double[] values, MiningFunction miningFunction, Schema schema){
		node.setId(String.valueOf(index));

		int featureIndex = features[index];

		// A non-leaf (binary split) node
		if(featureIndex >= 0){
			Feature feature = schema.getFeature(featureIndex);

			double threshold = thresholds[index];

			Predicate leftPredicate;
			Predicate rightPredicate;

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				if(threshold < 0 || threshold > 1){
					throw new IllegalArgumentException();
				}

				String value = binaryFeature.getValue();

				leftPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.NOT_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.EQUAL, value);
			} else

			{
				ContinuousFeature continuousFeature = feature
					.toContinuousFeature(DataType.FLOAT) // First, cast from any numeric type (including numpy.float64) to numpy.float32
					.toContinuousFeature(DataType.DOUBLE); // Second, cast from numpy.float32 to numpy.float64

				String value = ValueUtil.formatValue(threshold);

				leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_OR_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_THAN, value);
			}

			int leftIndex = leftChildren[index];
			int rightIndex = rightChildren[index];

			Node leftChild = new Node()
				.setPredicate(leftPredicate);

			encodeNode(leftChild, predicateManager, leftIndex, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

			Node rightChild = new Node()
				.setPredicate(rightPredicate);

			encodeNode(rightChild, predicateManager, rightIndex, leftChildren, rightChildren, features, thresholds, values, miningFunction, schema);

			node.addNodes(leftChild, rightChild);
		} else

		// A leaf node
		{
			if((MiningFunction.CLASSIFICATION).equals(miningFunction)){
				CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

				double[] scoreRecordCounts = getRow(values, leftChildren.length, categoricalLabel.size(), index);

				double recordCount = 0;

				for(double scoreRecordCount : scoreRecordCounts){
					recordCount += scoreRecordCount;
				}

				node.setRecordCount(recordCount);

				String score = null;

				Double probability = null;

				for(int i = 0; i < categoricalLabel.size(); i++){
					String value = categoricalLabel.getValue(i);

					ScoreDistribution scoreDistribution = new ScoreDistribution(value, scoreRecordCounts[i]);

					node.addScoreDistributions(scoreDistribution);

					double scoreProbability = (scoreRecordCounts[i] / recordCount);

					if(probability == null || probability.compareTo(scoreProbability) < 0){
						score = scoreDistribution.getValue();

						probability = scoreProbability;
					}
				}

				node.setScore(score);
			} else

			if((MiningFunction.REGRESSION).equals(miningFunction)){
				String score = ValueUtil.formatValue(values[index]);

				node.setScore(score);
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	static
	public Schema toTreeModelSchema(final DataType dataType, Schema schema){
		Function<Feature, Feature> function = new Function<Feature, Feature>(){

			@Override
			public Feature apply(Feature feature){

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					return binaryFeature;
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature(dataType);

					return continuousFeature;
				}
			}
		};

		return schema.toTransformedSchema(function);
	}

	static
	private double[] getRow(double[] values, int rows, int columns, int row){

		if(values.length != (rows * columns)){
			throw new IllegalArgumentException("Expected " + (rows * columns) + " element(s), got " + values.length + " element(s)");
		}

		double[] result = new double[columns];

		System.arraycopy(values, (row * columns), result, 0, columns);

		return result;
	}
}
