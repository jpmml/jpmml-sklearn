/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sktree.tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.tree.Tree;

public class ObliqueTree extends Tree {

	public ObliqueTree(String module, String name){
		super(module, name);
	}

	public ObliqueTree(ObliqueTree that){
		this(that.getPythonModule(), that.getPythonName());

		update(that);
	}

	public ObliqueTree transform(Schema schema){
		List<? extends Feature> features = schema.getFeatures();

		int[] featureIndices = new int[features.size()];

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			featureIndices[i] = (feature != null ? i : -2);
		}

		ObliqueTree result = new ObliqueTree(this){

			{
				delProjVecs();
			}

			@Override
			public int[] getFeature(){
				return featureIndices;
			}
		};

		return result;
	}

	public Schema transformSchema(Object segmentId, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		features = encodeFeatures(segmentId, (List)features, encoder);

		return new Schema(encoder, label, features);
	}

	public List<Feature> encodeFeatures(Object segmentId, List<Feature> features, SkLearnEncoder encoder){
		Integer nodeCount = getNodeCount();
		List<Number> projVecs = getProjVecs();

		int rows = nodeCount;
		int columns = features.size();

		Map<List<Number>, Feature> lcFeatures = new LinkedHashMap<>();

		List<Feature> result = new ArrayList<>();

		for(int row = 0; row < rows; row++){
			List<Number> weights = CMatrixUtil.getRow(projVecs, rows, columns, row);

			Feature feature;

			if(lcFeatures.containsKey(weights)){
				feature = lcFeatures.get(weights);
			} else

			{
				String name;

				if(segmentId != null){
					name = FieldNameUtil.create("lc", segmentId, row);
				} else

				{
					name = FieldNameUtil.create("lc", row);
				}

				feature = encodeFeature(name, features, weights, encoder);

				lcFeatures.put(weights, feature);
			}

			result.add(feature);
		}

		return result;
	}

	private Feature encodeFeature(String name, List<Feature> features, List<Number> weights, SkLearnEncoder encoder){
		ClassDictUtil.checkSize(features, weights);

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.SUM);

		Map<String, Feature> originalFeatures = new HashMap<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Number weight = weights.get(i);

			if(ValueUtil.isZero(weight)){
				continue;
			}

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Expression expression = continuousFeature.ref();
			if(!ValueUtil.isOne(weight)){
				expression = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, ExpressionUtil.createConstant(weight), expression);
			} else

			{
				FieldRef fieldRef = (FieldRef)expression;

				originalFeatures.put(fieldRef.requireField(), feature);
			}

			apply.addExpressions(expression);
		}

		List<Expression> expressions = apply.getExpressions();

		if(expressions.isEmpty()){
			return null;
		}

		Expression expression = apply;

		if(expressions.size() == 1){
			expression = Iterables.getOnlyElement(expressions);

			if(expression instanceof FieldRef){
				FieldRef fieldRef = (FieldRef)expression;

				return originalFeatures.get(fieldRef.requireField());
			}
		}

		DerivedField derivedField = encoder.createDerivedField(name, OpType.CONTINUOUS, DataType.FLOAT, expression);

		return new ContinuousFeature(encoder, derivedField);
	}

	public Integer getNodeCount(){
		return getInteger("node_count");
	}

	public boolean hasProjVecs(){
		return hasattr("proj_vecs");
	}

	public List<Number> getProjVecs(){
		return getNumberArray("proj_vecs");
	}

	public void delProjVecs(){
		delattr("proj_vecs");
	}
}