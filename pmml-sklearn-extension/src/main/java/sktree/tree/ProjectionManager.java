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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class ProjectionManager {

	private Map<List<Vector>, Feature> projections = new LinkedHashMap<>();


	public ProjectionManager(){
	}

	public Feature getOrCreateFeature(String name, List<Feature> features, List<Number> weights, SkLearnEncoder encoder){
		ClassDictUtil.checkSize(features, weights);

		List<Vector> key = createKey(features, weights);

		if(key.isEmpty()){
			return null;
		} // End if

		if(this.projections.containsKey(key)){
			return this.projections.get(key);
		} else

		{
			Feature feature = encodeFeature(name, key, encoder);

			this.projections.put(key, feature);

			return feature;
		}
	}

	static
	private Feature encodeFeature(String name, List<Vector> key, SkLearnEncoder encoder){
		List<Expression> plusExpressions = new ArrayList<>();
		List<Expression> minusExpressions = new ArrayList<>();

		for(int i = 0; i < key.size(); i++){
			Vector vector = key.get(i);

			Feature feature = vector.getFeature();
			Number weight = vector.getWeight();

			// XXX
			if(key.size() == 1){

				if(weight.doubleValue() == 1d){
					return feature;
				}
			}

			Expression expression;

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				expression = new NormDiscrete(binaryFeature.getName(), binaryFeature.getValue());
			} else

			{
				ContinuousFeature continuousFeature = feature.toContinuousFeature();

				expression = continuousFeature.ref();
			} // End if

			if(weight.doubleValue() == 1d){
				plusExpressions.add(expression);
			} else

			if(weight.doubleValue() == -1d){
				minusExpressions.add(expression);
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		Expression plusExpression = aggregate(plusExpressions);
		Expression minusExpression = aggregate(minusExpressions);

		Expression expression;

		if(plusExpression != null){

			if(minusExpression != null){
				expression = ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, plusExpression, minusExpression);
			} else

			{
				expression = plusExpression;
			}
		} else

		{
			if(minusExpression != null){
				expression = ExpressionUtil.toNegative(minusExpression);
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		DerivedField derivedField = encoder.createDerivedField(name, OpType.CONTINUOUS, DataType.FLOAT, expression);

		return new ContinuousFeature(encoder, derivedField);
	}

	static
	private Expression aggregate(List<Expression> expressions){

		if(expressions.isEmpty()){
			return null;
		} // End if

		if(expressions.size() == 1){
			return Iterables.getOnlyElement(expressions);
		} else

		{
			Apply apply = ExpressionUtil.createApply(PMMLFunctions.SUM);

			(apply.getExpressions()).addAll(expressions);

			return apply;
		}
	}

	static
	private List<Vector> createKey(List<Feature> features, List<Number> weights){
		List<Vector> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Number weight = weights.get(i);

			if(ValueUtil.isZero(weight)){
				continue;
			}

			result.add(new Vector(feature, weight));
		}

		return result;
	}

	static
	private class Vector {

		private Feature feature = null;

		private Number weight = null;


		private Vector(Feature feature, Number weight){
			setFeature(feature);
			setWeight(weight);
		}

		public Feature getFeature(){
			return this.feature;
		}

		public void setFeature(Feature feature){
			this.feature = Objects.requireNonNull(feature);
		}

		public Number getWeight(){
			return this.weight;
		}

		private void setWeight(Number weight){
			this.weight = Objects.requireNonNull(weight);
		}

		@Override
		public int hashCode(){
			int result = 0;

			result += (31 * result) + Objects.hashCode(this.getFeature());
			result += (31 * result) + Objects.hashCode(this.getWeight());

			return result;
		}

		@Override
		public boolean equals(Object object){

			if(object instanceof Vector){
				Vector that = (Vector)object;

				return Objects.equals(this.getFeature(), that.getFeature()) && Objects.equals(this.getWeight(), that.getWeight());
			}

			return false;
		}
	}
}