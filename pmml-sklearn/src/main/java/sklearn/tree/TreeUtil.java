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

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.google.common.primitives.Doubles;
import numpy.core.ScalarUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.dmg.pmml.HasExtensions;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMMLConstants;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.BranchNode;
import org.dmg.pmml.tree.ClassifierNode;
import org.dmg.pmml.tree.ComplexNode;
import org.dmg.pmml.tree.LeafNode;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.NodeTransformer;
import org.dmg.pmml.tree.SimplifyingNodeTransformer;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.CategoryManager;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ConversionException;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FloatUtil;
import org.jpmml.converter.MissingValueFeature;
import org.jpmml.converter.ModelTransformer;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.PayloadManager;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ThresholdFeature;
import org.jpmml.converter.ThresholdFeatureUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.tree.TreeModelTransformerBuilder;
import org.jpmml.converter.tree.visitors.TreeModelTransformer;
import org.jpmml.converter.visitors.AbstractExtender;
import org.jpmml.model.UnsupportedElementException;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnException;
import sklearn.Estimator;
import sklearn.EstimatorCastException;
import sklearn.HasApplyField;
import sklearn.HasMultiApplyField;
import sklearn.StepUtil;
import sklearn.VersionUtil;

public class TreeUtil {

	private TreeUtil(){
	}

	static
	public <E extends Estimator & HasTree> boolean hasMissingValueSupport(E estimator){
		Boolean sklearnAllowNaN = (Boolean)StepUtil.getTag(estimator.getInputTags(), "allow_nan");
		if(sklearnAllowNaN != null){
			return sklearnAllowNaN;
		}

		String sklearnVersion = estimator.getSkLearnVersion();
		if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, "1.3.0") >= 0){
			return true;
		}

		return false;
	}

	static
	public <E extends Estimator & HasTree> TreeModel encodeTreeModel(E estimator, MiningFunction miningFunction, PredicateManager predicateManager, PayloadManager payloadManager, Schema schema){
		Tree tree = estimator.getTree();
		boolean hasMissingValueSupport = estimator.hasMissingValueSupport();

		int[] leftChildren = tree.getChildrenLeft();
		int[] rightChildren = tree.getChildrenRight();
		int[] features = tree.getFeature();
		double[] thresholds = tree.getThreshold();
		int numberOfOutputs = estimator.getNumberOfOutputs();

		double[] values = tree.getValues();
		int[] missingGoToLeft = hasMissingValueSupport ? tree.getMissingGoToLeft() : null;

		Node root = encodeNode(0, True.INSTANCE, miningFunction, numberOfOutputs, leftChildren, rightChildren, features, thresholds, values, missingGoToLeft, new CategoryManager(), predicateManager, payloadManager, schema);

		TreeModel treeModel = new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT)
			.setMissingValueStrategy(hasMissingValueSupport ? TreeModel.MissingValueStrategy.DEFAULT_CHILD : TreeModel.MissingValueStrategy.NULL_PREDICTION);

		ClassDictUtil.clearContent(tree);

		return treeModel;
	}

	static
	private Node encodeNode(int index, Predicate predicate, MiningFunction miningFunction, int numberOfOutputs, int[] leftChildren, int[] rightChildren, int[] features, double[] thresholds, double[] values, int[] missingGoToLeft, CategoryManager categoryManager, PredicateManager predicateManager, PayloadManager payloadManager, Schema schema){
		Integer id = Integer.valueOf(index);

		int featureIndex = features[index];

		// A non-leaf (binary split) node
		if(featureIndex >= 0){
			Feature feature = schema.getFeature(featureIndex);

			double threshold = thresholds[index];

			CategoryManager leftCategoryManager = categoryManager;
			CategoryManager rightCategoryManager = categoryManager;

			Predicate leftPredicate;
			Predicate rightPredicate;

			Boolean defaultLeft = null;

			if(missingGoToLeft != null){
				defaultLeft = (missingGoToLeft[index] == 1);
			} // End if

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				if(threshold < 0 || threshold > 1){
					throw new SkLearnException("Expected [0, 1] range threshold value, got " + threshold);
				}

				Object value = binaryFeature.getValue();

				leftPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.NOT_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(binaryFeature, SimplePredicate.Operator.EQUAL, value);

				// XXX
				if(missingGoToLeft != null){
					defaultLeft = Boolean.TRUE;
				}
			} else

			if(feature instanceof MissingValueFeature){
				MissingValueFeature missingValueFeature = (MissingValueFeature)feature;

				if(threshold != 0.5d){
					throw new SkLearnException("Expected 0.5 threshold value, got " + threshold);
				}

				leftPredicate = predicateManager.createSimplePredicate(missingValueFeature, SimplePredicate.Operator.IS_NOT_MISSING, null);
				rightPredicate = predicateManager.createSimplePredicate(missingValueFeature, SimplePredicate.Operator.IS_MISSING, null);
			} else

			if(feature instanceof ThresholdFeature){
				ThresholdFeature thresholdFeature = (ThresholdFeature)feature;

				String name = thresholdFeature.getName();

				Object missingValue = thresholdFeature.getMissingValue();

				java.util.function.Predicate<Object> valueFilter = categoryManager.getValueFilter(name);

				if(!ValueUtil.isNaN(missingValue)){
					valueFilter = valueFilter.and(value -> !ValueUtil.isNaN(value));
				}

				float floatThreshold = toFloatThreshold(threshold);

				List<Object> leftValues = thresholdFeature.getValues((Number value) -> (value.floatValue() <= floatThreshold)).stream()
					.filter(valueFilter)
					.collect(Collectors.toList());

				List<Object> rightValues = thresholdFeature.getValues((Number value) -> (value.floatValue() > floatThreshold)).stream()
					.filter(valueFilter)
					.collect(Collectors.toList());

				leftCategoryManager = leftCategoryManager.fork(name, leftValues);
				rightCategoryManager = rightCategoryManager.fork(name, rightValues);

				leftPredicate = ThresholdFeatureUtil.createPredicate(thresholdFeature, leftValues, missingValue, predicateManager);
				rightPredicate = ThresholdFeatureUtil.createPredicate(thresholdFeature, rightValues, missingValue, predicateManager);
			} else

			{
				ContinuousFeature continuousFeature = feature.toContinuousFeature(DataType.FLOAT);

				Object value;

				// Happens when missing values are allowed
				if(threshold == Double.POSITIVE_INFINITY){
					value = PMMLConstants.POSITIVE_INFINITY;
				} else

				{
					value = toFloatThreshold(threshold);
				}

				leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_OR_EQUAL, value);
				rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_THAN, value);
			}

			int leftIndex = leftChildren[index];
			int rightIndex = rightChildren[index];

			Node leftChild = encodeNode(leftIndex, leftPredicate, miningFunction, numberOfOutputs, leftChildren, rightChildren, features, thresholds, values, missingGoToLeft, leftCategoryManager, predicateManager, payloadManager, schema);
			Node rightChild = encodeNode(rightIndex, rightPredicate, miningFunction, numberOfOutputs, leftChildren, rightChildren, features, thresholds, values, missingGoToLeft, rightCategoryManager, predicateManager, payloadManager, schema);

			Node result;

			if(miningFunction == MiningFunction.CLASSIFICATION){
				result = new ClassifierNode(null, predicate);
			} else

			if(miningFunction == MiningFunction.REGRESSION){
				double value = values[index];

				result = new BranchNode(value, predicate);
			} else

			{
				throw new IllegalArgumentException();
			}

			result
				.setId(id)
				.addNodes(leftChild, rightChild);

			if(defaultLeft != null){
				result.setDefaultChild(defaultLeft ? leftChild.getId() : rightChild.getId());
			}

			return result;
		} else

		// A leaf node
		{
			Node result;

			if(miningFunction == MiningFunction.CLASSIFICATION){
				CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();

				double[] leafValues = getRow(values, leftChildren.length, categoricalLabel.size(), index);

				List<Number> recordCounts = new AbstractList<Number>(){

					@Override
					public int size(){
						return leafValues.length;
					}

					@Override
					public Number get(int index){
						double leafValue = leafValues[index];

						return ValueUtil.narrow(leafValue);
					}
				};

				double totalRecordCount = 0d;

				for(Number recordCount : recordCounts){
					totalRecordCount += recordCount.doubleValue();
				}

				// XXX
				int maxIndex = PayloadManager.indexOfMax(Doubles.asList(leafValues));

				Object score = categoricalLabel.getValue(maxIndex);

				result = new ClassifierNode(score, predicate)
					.setId(id)
					.setRecordCount(ValueUtil.narrow(totalRecordCount));

				payloadManager.addScoreDistributions(result, recordCounts, null, categoricalLabel);
			} else

			if(miningFunction == MiningFunction.REGRESSION){

				if(numberOfOutputs == 1){
					double value = values[index];

					result = new LeafNode(value, predicate)
						.setId(id);
				} else

				{
					MultiLabel multiLabel = schema.requireMultiLabel()
						.expectSize(numberOfOutputs);

					double[] leafValues = getRow(values, leftChildren.length, numberOfOutputs, index);

					List<Number> scores = new AbstractList<Number>(){

						@Override
						public int size(){
							return leafValues.length;
						}

						@Override
						public Number get(int index){
							double leafValue = leafValues[index];

							return ValueUtil.narrow(leafValue);
						}
					};

					result = new ComplexNode(null, predicate)
						.setId(id);

					payloadManager.addScores(result, scores, multiLabel);
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			return result;
		}
	}

	static
	public <E extends Estimator & HasTreeOptions> Schema configureSchema(E estimator, Schema schema){
		Boolean numeric = (Boolean)estimator.getPMMLOption(HasTreeOptions.OPTION_NUMERIC, Boolean.TRUE);
		Boolean inputFloat = (Boolean)estimator.getPMMLOption(HasTreeOptions.OPTION_INPUT_FLOAT, Boolean.TRUE);

		return toTreeModelSchema(numeric, inputFloat, schema);
	}

	static
	public <E extends Estimator & HasTreeOptions, M extends Model> M configureModel(E estimator, M model){
		Boolean winnerId = (Boolean)estimator.getPMMLOption(HasTreeOptions.OPTION_WINNER_ID, Boolean.FALSE);

		if((Boolean.TRUE).equals(winnerId)){
			encodeNodeId(estimator, model);
		}

		List<String> names = new ArrayList<>();
		names.add(HasTreeOptions.OPTION_ALLOW_MISSING);
		names.addAll(HasTreeOptions.OPTIONS_TREEMODEL_TRANSFORMATIONS);
		names.addAll(HasTreeOptions.OPTIONS_NODE);

		Map<String, ?> options = estimator.getPMMLOptions(names);

		Set<MiningFunction> miningFunctions = EnumSet.noneOf(MiningFunction.class);

		Visitor miningFunctionCollector = new AbstractVisitor(){

			@Override
			public VisitorAction visit(TreeModel treeModel){
				miningFunctions.add(treeModel.requireMiningFunction());

				return super.visit(treeModel);
			}
		};
		miningFunctionCollector.applyTo(model);

		for(MiningFunction miningFunction : miningFunctions){
			ModelTransformer modelTransformer = createTreeModelTransformer(miningFunction, options);

			modelTransformer.applyTo(model);
		}

		return model;
	}

	static
	Schema toTreeModelSchema(Boolean numeric, Boolean inputFloat, Schema schema){
		Function<Feature, Feature> function = new Function<Feature, Feature>(){

			@Override
			public Feature apply(Feature feature){

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					return binaryFeature;
				} else

				if(feature instanceof MissingValueFeature){
					MissingValueFeature missingValueFeature = (MissingValueFeature)feature;

					return missingValueFeature;
				} else

				if(feature instanceof ThresholdFeature && (numeric != null && !numeric)){
					ThresholdFeature thresholdFeature = (ThresholdFeature)feature;

					return thresholdFeature;
				} else

				if(inputFloat != null && inputFloat){
					ContinuousFeature continuousFeature = feature.toContinuousFeature();

					DataType dataType = continuousFeature.getDataType();
					if(dataType != DataType.FLOAT){
						Field<?> field = continuousFeature.getField();

						field = FloatUtil.narrow(field);

						return new ContinuousFeature(continuousFeature.getEncoder(), field);
					}

					return continuousFeature;
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature(DataType.FLOAT);

					return continuousFeature;
				}
			}
		};

		return schema.toTransformedSchema(function);
	}

	static
	Schema toTreeModelFeatureImportanceSchema(Schema schema){
		Function<Feature, Feature> function = new Function<Feature, Feature>(){

			@Override
			public Feature apply(Feature feature){

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					return binaryFeature;
				} else

				if(feature instanceof MissingValueFeature){
					MissingValueFeature missingValueFeature = (MissingValueFeature)feature;

					return missingValueFeature;
				} else

				if(feature instanceof ThresholdFeature){
					ThresholdFeature thresholdFeature = (ThresholdFeature)feature;

					return thresholdFeature;
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature(DataType.FLOAT);

					return continuousFeature;
				}
			}
		};

		return schema.toTransformedSchema(function);
	}

	/**
	 * Undoes the &quot;outer layer&quot; of Scikit-Learn's <code>(double)(float)x</code> cast operation.
	 */
	static
	private float toFloatThreshold(double threshold){
		return FloatUtil.floor(threshold);
	}

	static
	private double[] getRow(double[] values, int rows, int columns, int row){

		if(values.length != (rows * columns)){
			throw new ConversionException("Expected " + ExceptionUtil.formatCount(rows * columns, "element") + ", got " + values.length);
		}

		double[] result = new double[columns];

		System.arraycopy(values, (row * columns), result, 0, columns);

		return result;
	}

	static
	private void encodeNodeId(Estimator estimator, Model model){

		if(model instanceof TreeModel){
			TreeModel treeModel = (TreeModel)model;

			estimator.encodeApplyOutput(treeModel, DataType.INTEGER);
		} else

		if(model instanceof MiningModel){
			MiningModel miningModel = (MiningModel)model;

			Segmentation segmentation = miningModel.requireSegmentation();

			List<Segment> segments = segmentation.requireSegments();

			List<String> segmentIds = new ArrayList<>();

			for(Segment segment : segments){
				@SuppressWarnings("unused")
				TreeModel treeModel = segment.requireModel(TreeModel.class);

				String segmentId = segment.getId();
				if(segmentId == null){
					throw new UnsupportedElementException(segment);
				}

				segmentIds.add(segmentId);
			}

			estimator.encodeMultiApplyOutput(miningModel, DataType.INTEGER, segmentIds);
		} else

		{
			throw new EstimatorCastException(estimator, Arrays.asList(HasApplyField.class, HasMultiApplyField.class));
		}
	}

	static
	private ModelTransformer createTreeModelTransformer(MiningFunction miningFunction, Map<String, ?> options){
		ModelTransformer result = new TreeModelTransformerBuilder(miningFunction){

			@Override
			public ModelTransformer build(){
				ModelTransformer result = super.build();

				List<Visitor> visitors = new ArrayList<>();

				if(!isEnabled(HasTreeOptions.OPTION_ALLOW_MISSING)){
					visitors.add(createNodeDefaultChildCleaner(miningFunction));
				} // End if

				if(isEnabled(HasTreeOptions.OPTION_NODE_EXTENSIONS)){
					Map<String, Map<Integer, ?>> nodeExtensions = (Map)getOption(HasTreeOptions.OPTION_NODE_EXTENSIONS);

					Collection<? extends Map.Entry<String, Map<Integer, ?>>> entries = nodeExtensions.entrySet();

					for(Map.Entry<String, Map<Integer, ?>> entry : entries){
						visitors.add(createNodeExtender(miningFunction, entry.getKey(), entry.getValue()));
					}
				} // End if

				if(isDisabled(HasTreeOptions.OPTION_NODE_SCORE)){
					visitors.add(createNodeScoreCleaner(miningFunction));
				}

				result.addAll(0, visitors);

				return result;
			}

			@Override
			protected Map<String, ?> getDefaults(){
				Map<String, Object> result = new LinkedHashMap<>();
				result.put(HasTreeOptions.OPTION_COMPACT, Boolean.TRUE);
				result.put(HasTreeOptions.OPTION_PRUNE, Boolean.TRUE);

				if(isExplicitlyEnabled(HasTreeOptions.OPTION_WINNER_ID)){
					result.put(HasTreeOptions.OPTION_NODE_ID, Boolean.TRUE);
					result.put(HasTreeOptions.OPTION_NODE_SCORE, Boolean.TRUE);
				} else

				if(!isExplicitlyEnabled(HasTreeOptions.OPTION_ALLOW_MISSING)){
					result.put(HasTreeOptions.OPTION_NODE_ID, Boolean.FALSE);
				}

				return result;
			}

			@Override
			protected List<String> getNodeOptions(){
				return HasTreeOptions.OPTIONS_NODE;
			}
		}
			.putOptions(options)
			.build();

		return result;
	}

	static
	private Visitor createNodeDefaultChildCleaner(MiningFunction miningFunction){
		Visitor result = new TreeModelTransformer(miningFunction){

			@Override
			public void enterNode(Node node){
				clearDefaultChild(node);
			}

			@Override
			public void exitNode(Node node){
			}

			@Override
			public void enterTreeModel(TreeModel treeModel){
				super.enterTreeModel(treeModel);

				treeModel.setMissingValueStrategy(null);
			}

			@Override
			public void exitTreeModel(TreeModel treeModel){
				super.exitTreeModel(treeModel);
			}
		};

		return result;
	}

	static
	private Visitor createNodeExtender(MiningFunction miningFunction, String name, Map<Integer, ?> values){
		Visitor result = new AbstractExtender(name){

			private NodeTransformer nodeTransformer = SimplifyingNodeTransformer.INSTANCE;


			@Override
			public VisitorAction visit(TreeModel treeModel){

				if(treeModel.requireMiningFunction() != miningFunction){
					return VisitorAction.SKIP;
				}

				treeModel.setNode(ensureExtensibility(treeModel.getNode()));

				return super.visit(treeModel);
			}

			@Override
			public VisitorAction visit(Node node){

				if(node.hasNodes()){
					List<Node> children = node.getNodes();

					for(ListIterator<Node> childIt = children.listIterator(); childIt.hasNext(); ){
						childIt.set(ensureExtensibility(childIt.next()));
					}
				}

				Object value = getValue(node);
				if(value != null){
					value = ScalarUtil.decode(value);

					addExtension((Node & HasExtensions)node, ValueUtil.asString(value));
				}

				return super.visit(node);
			}

			private Node ensureExtensibility(Node node){

				if(node instanceof HasExtensions){
					return node;
				}

				Object value = getValue(node);
				if(value != null){
					return this.nodeTransformer.toComplexNode(node);
				}

				return node;
			}

			private Object getValue(Node node){
				Integer id = ValueUtil.asInteger((Number)node.requireId());

				return values.get(id);
			}
		};

		return result;
	}

	static
	private Visitor createNodeScoreCleaner(MiningFunction miningFunction){
		Visitor result = new TreeModelTransformer(miningFunction){

			@Override
			public void enterNode(Node node){

				if(node.hasNodes()){
					clearScore(node);

					if(node.hasPayloads()){
						(node.getPayloads()).clear();
					}
				}
			}

			@Override
			public void exitNode(Node node){
			}
		};

		return result;
	}
}
