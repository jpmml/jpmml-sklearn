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
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import chaid.Column;
import com.google.common.math.DoubleMath;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Output;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.ClassifierNode;
import org.dmg.pmml.tree.CountingBranchNode;
import org.dmg.pmml.tree.CountingLeafNode;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import treelib.Node;
import treelib.Tree;

public class CHAIDUtil {

	private CHAIDUtil(){
	}

	static
	public TreeModel encodeModel(MiningFunction miningFunction, Tree tree, Schema schema){
		org.dmg.pmml.tree.Node root = encodeNode(True.INSTANCE, tree.selectRoot(), tree, new PredicateManager(), schema);

		Output output = new Output()
			.addOutputFields(ModelUtil.createEntityIdField("nodeId", DataType.INTEGER));

		TreeModel treeModel = new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema.getLabel()), root)
			.setOutput(output);

		return treeModel;
	}

	static
	private org.dmg.pmml.tree.Node encodeNode(Predicate predicate, Node node, Tree tree, PredicateManager predicateManager, Schema schema){
		org.dmg.pmml.tree.Node result;

		Label label = schema.getLabel();

		chaid.Node tag = node.getTag(chaid.Node.class);

		List<Node> successors = node.selectSuccessors(tree);

		Column depV = tag.getDepV();
		List<?> indices = tag.getIndices();
		chaid.Split split = tag.getSplit();

		List<? extends Number> depVArr = depV.getArr();

		ClassDictUtil.checkSize(depVArr, indices);

		Integer columnId = split.getColumnId();
		List<List<Integer>> splits = split.getSplits();

		ClassDictUtil.checkSize(successors, splits);

		if(!successors.isEmpty()){
			CategoricalFeature categoricalFeature = (CategoricalFeature)schema.getFeature(columnId);

			if(label instanceof CategoricalLabel){
				result = new ClassifierNode(null, predicate);
			} else

			{
				result = new CountingBranchNode(null, predicate);
			}

			for(int i = 0; i < successors.size(); i++){
				Node successor = successors.get(i);

				List<Object> values = new ArrayList<>();

				List<Integer> splitValues = splits.get(i);
				for(Integer splitValue : splitValues){
					values.add(categoricalFeature.getValue(splitValue));
				}

				Predicate successorPredicate = predicateManager.createPredicate(categoricalFeature, values);

				result.addNodes(encodeNode(successorPredicate, successor, tree, predicateManager, schema));
			}
		} else

		{
			if(label instanceof CategoricalLabel){
				result = new ClassifierNode(null, predicate);
			} else

			{
				result = new CountingLeafNode(null, predicate);
			}
		}

		result
			.setId(node.getIdentifier())
			.setRecordCount(depVArr.size());

		if(label instanceof ContinuousLabel){
			ContinuousLabel continuousLabel = (ContinuousLabel)label;

			Double score = DoubleMath.mean(depVArr);

			result.setScore(score);
		} else

		if(label instanceof CategoricalLabel){
			CategoricalLabel categoricalLabel = (CategoricalLabel)label;

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

				ScoreDistribution scoreDistribution = new ScoreDistribution(value, count);

				scoreDistributions.add(scoreDistribution);
			}
		} else

		{
			throw new IllegalArgumentException();
		}

		return result;
	}
}