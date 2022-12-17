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
import java.util.Collections;
import java.util.List;

import org.dmg.pmml.Apply;
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
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
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

		switch(dtype){
			case "numerical":
			case "categorical":
				break;
			default:
				throw new IllegalArgumentException(dtype);
		}

		List<Double> categoriesOut = getCategoriesOut();

		SchemaUtil.checkSize(1, features);

		Feature feature = features.get(0);

		Expression expression;

		List<Double> categories;

		if(!splits.isEmpty()){
			OptimalBinningUtil.checkIncreasingOrder(splits);

			switch(dtype){
				case "numerical":
					{
						expression = encodeNumericalBinning(feature, splits, categoriesOut);
					}
					break;
				case "categorical":
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

						expression = encodeCategoricalBinning(feature, splits, categoriesIn, categoriesOut);
					}
					break;
				default:
					throw new IllegalArgumentException(dtype);
			}

			categories = categoriesOut.subList(0, splits.size() + 2);
		} else

		{
			Expression apply = PMMLUtil.createApply(PMMLFunctions.IF,
				PMMLUtil.createApply(PMMLFunctions.ISNOTMISSING, feature.ref()),
				PMMLUtil.createConstant(categoriesOut.get(0), null),
				PMMLUtil.createConstant(0d)
			);

			expression = apply;

			categories = categoriesOut.subList(0, 1);
		} // End if

		if(!specialCodes.isEmpty()){
			// XXX
			Double categorySpecial = 0d;

			Apply valueApply = PMMLUtil.createApply((specialCodes.size() == 1 ? PMMLFunctions.EQUAL : PMMLFunctions.ISIN), feature.ref());

			for(Number specialCode : specialCodes){
				valueApply.addExpressions(PMMLUtil.createConstant(specialCode));
			}

			if(expression instanceof HasMapMissingTo){
				HasMapMissingTo<?, ?> hasMapMissingTo = (HasMapMissingTo<?, ?>)expression;

				valueApply.setMapMissingTo(hasMapMissingTo.getMapMissingTo());
			}

			Apply ifApply = PMMLUtil.createApply(PMMLFunctions.IF,
				valueApply,
				PMMLUtil.createConstant(categorySpecial),
				expression
			);

			expression = ifApply;

			categories = new ArrayList<>(categories);
			categories.add(categorySpecial);
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("optBinning", feature), OpType.CATEGORICAL, DataType.DOUBLE, expression);

		feature = new CategoricalFeature(encoder, derivedField, categories);

		return Collections.singletonList(feature);
	}

	private Discretize encodeNumericalBinning(Feature feature, List<Number> splits, List<Double> categoriesOut){
		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		Discretize discretize = new Discretize(continuousFeature.getName())
			.setMapMissingTo(0d);

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

			Interval interval = new Interval(Interval.Closure.CLOSED_OPEN)
				.setLeftMargin(leftMargin)
				.setRightMargin(rightMargin);

			DiscretizeBin discretizeBin = new DiscretizeBin(categoriesOut.get(i), interval);

			discretize.addDiscretizeBins(discretizeBin);
		}

		return discretize;
	}

	private MapValues encodeCategoricalBinning(Feature feature, List<Number> splits, List<?> categoriesIn, List<Double> categoriesOut){
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
		}

		MapValues mapValues = PMMLUtil.createMapValues(feature.getName(), inputValues, outputValues)
			.setMapMissingTo(0.0d);

		return mapValues;
	}

	public List<?> getCategoriesIn(){
		return getArray("_categories");
	}

	public List<Double> getCategoriesOut(){
		String metric = getMetric();
		List<Integer> numberOfEvents = getNumberOfEvents();
		List<Integer> numberOfNonEvents = getNumberOfNonEvents();

		switch(metric){
			case "event_rate":
			case "woe":
				break;
			default:
				throw new IllegalArgumentException();
		}

		ClassDictUtil.checkSize(numberOfEvents, numberOfNonEvents);

		double constant = Math.log((double)OptimalBinningUtil.sumExact(numberOfEvents) / (double)OptimalBinningUtil.sumExact(numberOfNonEvents));

		List<Double> result = new ArrayList<>();

		for(int i = 0; i < numberOfEvents.size(); i++){
			double eventRate = (double)numberOfEvents.get(i) / (double)Math.addExact(numberOfEvents.get(i), numberOfNonEvents.get(i));

			switch(metric){
				case "event_rate":
					{
						result.add(eventRate);
					}
					break;
				case "woe":
					{
						double woe = Math.log((1d / eventRate) - 1d) + constant;

						result.add(woe);
					}
					break;
				default:
					throw new IllegalArgumentException();
			}
		}

		return result;
	}

	public String getDType(){
		return getString("dtype");
	}

	public String getDefaultMetric(){
		return "woe";
	}

	public String getMetric(){

		if(!containsKey("metric")){
			return getDefaultMetric();
		}

		return getString("metric");
	}

	public OptimalBinning setMetric(String metric){
		put("metric", metric);

		return this;
	}

	public List<Integer> getNumberOfEvents(){
		return getIntegerArray("_n_event");
	}

	public List<Integer> getNumberOfNonEvents(){
		return getIntegerArray("_n_nonevent");
	}

	public List<Number> getSpecialCodes(){
		Object specialCodes = get("special_codes");

		if(specialCodes == null){
			return Collections.emptyList();
		}

		return getListLike("special_codes", Number.class);
	}

	public List<Number> getSplitsOptimal(){
		return getNumberArray("_splits_optimal");
	}
}