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
package optbinning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.dmg.pmml.Apply;
import org.dmg.pmml.CompoundPredicate;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.Expression;
import org.dmg.pmml.HasMapMissingTo;
import org.dmg.pmml.Interval;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class OptimalBinning extends Transformer {

	public OptimalBinning(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		String dtype = getDType();
		List<Number> specialCodes = getSpecialCodes();
		List<Number> splits = getSplitsOptimal();

		List<Double> categoriesOut = getCategoriesOut();

		ClassDictUtil.checkSize(splits.size() + 3, categoriesOut);

		// Drop the last two elements, which correspond to "special" and "missing" categories
		categoriesOut = categoriesOut.subList(0, categoriesOut.size() - 2);

		SchemaUtil.checkSize(1, features);

		Feature feature = features.get(0);

		Expression expression;

		PredicateManager predicateManager = new PredicateManager();

		List<Predicate> predicates = new ArrayList<>();

		if(!splits.isEmpty()){
			OptimalBinningUtil.checkIncreasingOrder(splits);

			switch(dtype){
				case OptimalBinning.DTYPE_NUMERICAL:
					{
						expression = encodeNumericalBinning(feature, splits, categoriesOut, predicateManager, predicates);
					}
					break;
				case OptimalBinning.DTYPE_CATEGORICAL:
					{
						List<?> categoriesIn = getCategoriesIn();

						if(feature instanceof WildcardFeature){
							WildcardFeature wildcardFeature = (WildcardFeature)feature;

							DataType dataType = TypeUtil.getDataType(categoriesIn, DataType.STRING);

							DataField dataField = wildcardFeature.getField();
							if(dataField.requireDataType() != dataType){
								dataField.setDataType(dataType);
							}

							feature = wildcardFeature.toCategoricalFeature(categoriesIn);
						}

						expression = encodeCategoricalBinning(feature, splits, categoriesIn, categoriesOut, predicateManager, predicates);
					}
					break;
				default:
					throw new IllegalArgumentException(dtype);
			}
		} else

		{
			Expression apply = ExpressionUtil.createApply(PMMLFunctions.IF,
				ExpressionUtil.createApply(PMMLFunctions.ISNOTMISSING, feature.ref()),
				ExpressionUtil.createConstant(null, categoriesOut.get(0)),
				ExpressionUtil.createConstant(OptimalBinning.CATEGORY_MISSING)
			);

			expression = apply;
		}

		List<Double> categories = categoriesOut.stream()
			.distinct()
			.collect(Collectors.toList());

		// Special
		if(!specialCodes.isEmpty()){
			Apply valueApply = ExpressionUtil.createValueApply(feature.ref(), specialCodes);

			if(expression instanceof HasMapMissingTo){
				HasMapMissingTo<?, ?> hasMapMissingTo = (HasMapMissingTo<?, ?>)expression;

				valueApply.setMapMissingTo(hasMapMissingTo.getMapMissingTo());
			}

			expression = ExpressionUtil.createApply(PMMLFunctions.IF,
				valueApply,
				ExpressionUtil.createConstant(OptimalBinning.CATEGORY_SPECIAL),
				expression
			);

			Predicate specialPredicate = predicateManager.createPredicate(feature, specialCodes);

			predicates.add(specialPredicate);

			categories = OptimalBinningUtil.ensureCategory(categories, OptimalBinning.CATEGORY_SPECIAL);
		} else

		{
			predicates.add(null);
		}

		// Missing
		{
			Predicate missingPredicate = predicateManager.createSimplePredicate(feature, SimplePredicate.Operator.IS_MISSING, null);

			predicates.add(missingPredicate);

			categories = OptimalBinningUtil.ensureCategory(categories, OptimalBinning.CATEGORY_MISSING);
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("optBinning", feature), OpType.CATEGORICAL, DataType.DOUBLE, expression);

		feature = new BinnedFeature(encoder, derivedField, categories, predicates);

		return Collections.singletonList(feature);
	}

	private Discretize encodeNumericalBinning(Feature feature, List<Number> splits, List<Double> categoriesOut, PredicateManager predicateManager, List<Predicate> predicates){
		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		Discretize discretize = new Discretize(continuousFeature.getName())
			.setMapMissingTo(OptimalBinning.CATEGORY_MISSING);

		for(int i = 0; i <= splits.size(); i++){
			Number leftMargin = null;
			Number rightMargin = null;

			if(i == 0){
				rightMargin = splits.get(i);
			} else

			if(i == splits.size()){
				leftMargin = splits.get(i - 1);
			} else

			{
				leftMargin = splits.get(i - 1);
				rightMargin = splits.get(i);
			}

			Interval interval = new Interval(Interval.Closure.CLOSED_OPEN, leftMargin, rightMargin);

			DiscretizeBin discretizeBin = new DiscretizeBin(categoriesOut.get(i), interval);

			discretize.addDiscretizeBins(discretizeBin);

			Predicate leftPredicate = null;
			Predicate rightPredicate = null;

			if(leftMargin != null){
				leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_OR_EQUAL, leftMargin);
			} // End if

			if(rightMargin != null){
				rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_THAN, rightMargin);
			}

			Predicate predicate;

			if(leftPredicate != null && rightPredicate != null){
				predicate = predicateManager.createCompoundPredicate(CompoundPredicate.BooleanOperator.AND, leftPredicate, rightPredicate);
			} else

			{
				predicate = (leftPredicate != null ? leftPredicate : rightPredicate);
			}

			predicates.add(predicate);
		}

		return discretize;
	}

	private MapValues encodeCategoricalBinning(Feature feature, List<Number> splits, List<?> categoriesIn, List<Double> categoriesOut, PredicateManager predicateManager, List<Predicate> predicates){
		List<Object> inputValues = new ArrayList<>();
		List<Double> outputValues = new ArrayList<>();

		int begin = 0;

		for(int i = 0; i <= splits.size(); i++){
			Double splitCategoryOut = categoriesOut.get(i);

			int end;

			if(i < splits.size()){
				Number split = splits.get(i);

				end = (int)Math.ceil(split.doubleValue());
			} else

			{
				end = categoriesIn.size();
			}

			List<?> splitCategoriesIn = categoriesIn.subList(begin, end);

			for(Object splitCategoryIn : splitCategoriesIn){
				inputValues.add(splitCategoryIn);
				outputValues.add(splitCategoryOut);
			}

			begin = end;

			Predicate predicate = predicateManager.createPredicate(feature, splitCategoriesIn);

			predicates.add(predicate);
		}

		MapValues mapValues = ExpressionUtil.createMapValues(feature.getName(), inputValues, outputValues)
			.setMapMissingTo(OptimalBinning.CATEGORY_MISSING);

		return mapValues;
	}

	public List<Object> getCategoriesIn(){
		return getObjectArray("_categories");
	}

	public List<Double> getCategoriesOut(){
		String metric = getMetric();
		List<Integer> numberOfEvents = getNumberOfEvents();
		List<Integer> numberOfNonEvents = getNumberOfNonEvents();

		ClassDictUtil.checkSize(numberOfEvents, numberOfNonEvents);

		double constant = Math.log((double)OptimalBinningUtil.sumExact(numberOfEvents) / (double)OptimalBinningUtil.sumExact(numberOfNonEvents));

		List<Double> result = new ArrayList<>();

		for(int i = 0; i < numberOfEvents.size(); i++){
			double eventRate = (double)numberOfEvents.get(i) / (double)Math.addExact(numberOfEvents.get(i), numberOfNonEvents.get(i));

			switch(metric){
				case OptimalBinning.METRIC_EVENT_RATE:
					{
						result.add(eventRate);
					}
					break;
				case OptimalBinning.METRIC_WOE:
					{
						double woe = Math.log((1d / eventRate) - 1d) + constant;

						result.add(woe);
					}
					break;
				default:
					throw new IllegalArgumentException(metric);
			}
		}

		return result;
	}

	public String getDType(){
		return getEnum("dtype", this::getString, Arrays.asList(OptimalBinning.DTYPE_CATEGORICAL, OptimalBinning.DTYPE_NUMERICAL));
	}

	public String getDefaultMetric(){
		return OptimalBinning.METRIC_WOE;
	}

	public Collection<String> getSupportedMetrics(){
		return Arrays.asList(OptimalBinning.METRIC_EVENT_RATE, OptimalBinning.METRIC_WOE);
	}

	public String getMetric(){

		if(!hasattr("metric")){
			return getDefaultMetric();
		}

		return getEnum("metric", this::getString, getSupportedMetrics());
	}

	public OptimalBinning setMetric(String metric){
		setattr("metric", metric);

		return this;
	}

	public List<Integer> getNumberOfEvents(){
		return getIntegerArray("_n_event");
	}

	public List<Integer> getNumberOfNonEvents(){
		return getIntegerArray("_n_nonevent");
	}

	public List<Number> getSpecialCodes(){
		Object specialCodes = getOptionalObject("special_codes");

		if(specialCodes == null){
			return Collections.emptyList();
		}

		return getListLike("special_codes", Number.class);
	}

	public List<Number> getSplitsOptimal(){
		return getNumberArray("_splits_optimal");
	}

	public static final Double CATEGORY_MISSING = 0d;
	public static final Double CATEGORY_SPECIAL = 0d;

	private static final String DTYPE_CATEGORICAL = "categorical";
	private static final String DTYPE_NUMERICAL = "numerical";

	private static final String METRIC_EVENT_RATE = "event_rate";
	private static final String METRIC_WOE = "woe";
}