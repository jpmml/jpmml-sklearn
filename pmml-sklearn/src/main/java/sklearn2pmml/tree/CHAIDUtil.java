/*
 * Copyright (c) 2022 Villu Ruusmann
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
package sklearn2pmml.tree;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import chaid.Column;
import com.google.common.math.DoubleMath;
import org.dmg.pmml.CompoundPredicate;
import org.dmg.pmml.False;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.ScoreFrequency;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.ClassifierNode;
import org.dmg.pmml.tree.CountingBranchNode;
import org.dmg.pmml.tree.CountingLeafNode;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DiscreteFeature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;
import treelib.Node;
import treelib.Tree;

public class CHAIDUtil {

	private CHAIDUtil(){
	}

	static
	public <E extends Estimator & HasTree> TreeModel encodeModel(E estimator, MiningFunction miningFunction, Schema schema){
		Tree tree = estimator.getTree();

		org.dmg.pmml.tree.Node root = encodeNode(True.INSTANCE, tree.selectRoot(), tree, new PredicateManager(), schema);

		return new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema), root);
	}

	static
	private org.dmg.pmml.tree.Node encodeNode(Predicate predicate, Node node, Tree tree, PredicateManager predicateManager, Schema schema){
		org.dmg.pmml.tree.Node result;

		ScalarLabel scalarLabel = schema.requireScalarLabel();

		chaid.Node tag = node.getTag(chaid.Node.class);

		List<Node> successors = node.selectSuccessors(tree);

		Column depV = tag.getDepV();
		List<?> indices = tag.getIndices();
		chaid.Split split = tag.getSplit();

		List<? extends Number> depVArr = depV.getArr();

		ClassDictUtil.checkSize(depVArr, indices);

		Integer columnId = split.getColumnId();
		List<List<Integer>> splits = split.getSplits();
		List<List<Object>> splitMap = split.getSplitMap();

		ClassDictUtil.checkSize(successors, splits, splitMap);

		Comparator<Node> comparator = new Comparator<Node>(){

			@Override
			public int compare(Node left, Node right){
				chaid.Node leftTag = left.getTag(chaid.Node.class);
				chaid.Node rightTag = right.getTag(chaid.Node.class);

				List<?> leftIndices = leftTag.getIndices();
				List<?> rightIndices = rightTag.getIndices();

				return Integer.compare(leftIndices.size(), rightIndices.size());
			}
		};

		if(!successors.isEmpty()){
			DiscreteFeature discreteFeatue = (DiscreteFeature)schema.getFeature(columnId);

			Collection<?> categories = discreteFeatue.getValues();

			if(scalarLabel instanceof CategoricalLabel){
				result = new ClassifierNode(null, predicate);
			} else

			{
				result = new CountingBranchNode(null, predicate);
			}

			Set<Object> unusedValues = new LinkedHashSet<>(categories);

			for(int i = 0; i < successors.size(); i++){
				List<Integer> splitIndices = splits.get(i);
				List<?> splitValues = splitMap.get(i);

				ClassDictUtil.checkSize(splitIndices, splitValues);

				for(int j = 0; j < splitIndices.size(); j++){
					Integer splitIndex = splitIndices.get(j);
					Object splitValue = splitValues.get(j);

					if(isMissing(splitIndex, splitValue)){
						// Ignored
					} else

					{
						removeCategory(unusedValues, splitValue);
					}
				}
			}

			// The node with the most training data records
			Node maxSuccessor = null;

			if(!unusedValues.isEmpty()){

				for(int i = 0; i < successors.size(); i++){
					Node successor = successors.get(i);

					if(maxSuccessor == null || comparator.compare(successor, maxSuccessor) >= 0){
						maxSuccessor = successor;
					}
				}
			}

			for(int i = 0; i < successors.size(); i++){
				Node successor = successors.get(i);

				List<Integer> splitIndices = splits.get(i);
				List<?> splitValues = splitMap.get(i);

				List<Object> values = new ArrayList<>();

				boolean withMissing = false;

				for(int j = 0; j < splitIndices.size(); j++){
					Integer splitIndex = splitIndices.get(j);
					Object splitValue = splitValues.get(j);

					if(isMissing(splitIndex, splitValue)){
						withMissing = true;
					} else

					{
						Object value = selectCategory(categories, splitValue);

						values.add(value);
					}
				}

				if(Objects.equals(successor, maxSuccessor)){
					values.addAll(unusedValues);
				}

				Predicate successorPredicate;

				if(!values.isEmpty()){
					successorPredicate = predicateManager.createPredicate(discreteFeatue, values);

					if(withMissing){
						successorPredicate = predicateManager.createCompoundPredicate(CompoundPredicate.BooleanOperator.SURROGATE,
							successorPredicate,
							predicateManager.createSimplePredicate(discreteFeatue, SimplePredicate.Operator.IS_MISSING, null)
						);
					}
				} else

				{
					successorPredicate = False.INSTANCE;

					if(withMissing){
						successorPredicate = predicateManager.createSimplePredicate(discreteFeatue, SimplePredicate.Operator.IS_MISSING, null);
					}
				}

				result.addNodes(encodeNode(successorPredicate, successor, tree, predicateManager, schema));
			}
		} else

		{
			if(scalarLabel instanceof CategoricalLabel){
				result = new ClassifierNode(null, predicate);
			} else

			{
				result = new CountingLeafNode(null, predicate);
			}
		}

		result
			.setId(node.getIdentifier())
			.setRecordCount(depVArr.size());

		if(scalarLabel instanceof ContinuousLabel){
			ContinuousLabel continuousLabel = (ContinuousLabel)scalarLabel;

			Double score = DoubleMath.mean(depVArr);

			result.setScore(score);
		} else

		if(scalarLabel instanceof CategoricalLabel){
			CategoricalLabel categoricalLabel = (CategoricalLabel)scalarLabel;

			Map<Integer, Long> countMap = ((List<Integer>)depVArr).stream()
				.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

			List<ScoreDistribution> scoreDistributions = result.getScoreDistributions();

			Long maxCount = null;

			Collection<? extends Map.Entry<Integer, Long>> entries = countMap.entrySet();
			for(Map.Entry<Integer, Long> entry : entries){
				Object value = categoricalLabel.getValue(entry.getKey());
				Long count = entry.getValue();

				if(maxCount == null || (maxCount).compareTo(count) < 0){
					maxCount = count;

					result.setScore(value);
				}

				ScoreDistribution scoreDistribution = new ScoreFrequency(value, count);

				scoreDistributions.add(scoreDistribution);
			}
		} else

		{
			throw new IllegalArgumentException();
		}

		return result;
	}

	static
	private boolean isMissing(Integer splitIndex, Object splitValue){

		// Floating-point data type columns
		if(splitIndex == -1){
			return true;
		} else

		// Object data type columns
		if(splitValue == null){
			return true;
		}

		return false;
	}

	static
	private void removeCategory(Collection<?> values, Object splitValue){

		for(Iterator<?> it = values.iterator(); it.hasNext(); ){
			Object value = it.next();

			boolean matches = equals(value, splitValue);
			if(matches){
				it.remove();

				return;
			}
		}

		throw new IllegalArgumentException();
	}

	static
	private Object selectCategory(Collection<?> values, Object splitValue){

		for(Iterator<?> it = values.iterator(); it.hasNext(); ){
			Object value = it.next();

			boolean matches = equals(value, splitValue);
			if(matches){
				return value;
			}
		}

		throw new IllegalArgumentException();
	}

	static
	private boolean equals(Object left, Object right){

		if((left instanceof Number) && (right instanceof Number)){
			return (Double.compare(((Number)left).doubleValue(), ((Number)right).doubleValue()) == 0);
		}

		return Objects.equals(left, right);
	}
}