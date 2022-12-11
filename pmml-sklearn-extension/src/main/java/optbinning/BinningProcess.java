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
import java.util.Map;

import com.google.common.collect.Maps;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Interval;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Initializer;
import sklearn.InitializerUtil;

public class BinningProcess extends Initializer {

	public BinningProcess(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> initializeFeatures(SkLearnEncoder encoder){
		return encodeFeatures(Collections.emptyList(), encoder);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<Boolean> support = getSupport();

		Map<String, OptimalBinning> binnedVariables = getBinnedVariables();

		List<String> variableNames = getVariableNames();
		Map<String, String> variableDTypes = getVariableDTypes();
		Map<String, Map<String, ?>> variableStats = getVariableStats();

		ClassDictUtil.checkSize(support, variableNames, variableDTypes.values(), variableStats.values());

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < support.size(); i++){
			boolean flag = support.get(i);

			if(!flag){
				continue;
			}

			String variableName = variableNames.get(i);

			OptimalBinning optimalBinning = binnedVariables.get(variableName);

			String dtype = optimalBinning.getDType();
			switch(dtype){
				case "numerical":
					break;
				default:
					throw new IllegalArgumentException(dtype);
			}

			List<Number> splits = optimalBinning.getSplitsOptimal();
			List<? extends Number> categories = optimalBinning.getCategories();

			Feature feature = InitializerUtil.selectFeature(variableName, features, encoder);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			if(!splits.isEmpty()){
				OptimalBinningUtil.checkIncreasingOrder(splits);

				Discretize discretize = new Discretize(continuousFeature.getName())
					.setMapMissingTo(0d);

				for(int j = 0; j <= splits.size(); j++){
					Number leftMargin = null;
					Number rightMargin = null;

					if(j == 0){
						rightMargin = splits.get(j);
					} else

					if(j == splits.size()){
						leftMargin = splits.get(j - 1);
					} else

					{
						leftMargin = splits.get(j - 1);
						rightMargin = splits.get(j);
					}

					Interval interval = new Interval(Interval.Closure.CLOSED_OPEN)
						.setLeftMargin(leftMargin)
						.setRightMargin(rightMargin);

					DiscretizeBin discretizeBin = new DiscretizeBin(categories.get(j), interval);

					discretize.addDiscretizeBins(discretizeBin);
				}

				DerivedField derivedField = encoder.createDerivedField(createFieldName("optBinning", continuousFeature), OpType.CATEGORICAL, DataType.DOUBLE, discretize);

				feature = new CategoricalFeature(encoder, derivedField, categories.subList(0, splits.size() + 2));
			} else

			{
				Expression expression = PMMLUtil.createApply(PMMLFunctions.IF,
					PMMLUtil.createApply(PMMLFunctions.ISNOTMISSING, continuousFeature.ref()),
					PMMLUtil.createConstant(categories.get(0), null),
					PMMLUtil.createConstant(0d)
				);

				DerivedField derivedField = encoder.createDerivedField(createFieldName("optBinning", continuousFeature), OpType.CATEGORICAL, DataType.DOUBLE, expression);

				feature = new CategoricalFeature(encoder, derivedField, categories.subList(0, 1));
			}

			result.add(feature);
		}

		return result;
	}

	public Map<String, OptimalBinning> getBinnedVariables(){
		Map<String, ?> binnedVariables = getDict("_binned_variables");

		CastFunction<OptimalBinning> castFunction = new CastFunction<OptimalBinning>(OptimalBinning.class){

			@Override
			protected String formatMessage(Object object){
				return "The binning object (" + ClassDictUtil.formatClass(object) + ") is not a supported";
			}
		};

		return Maps.transformValues(binnedVariables, castFunction);
	}

	public List<Boolean> getSupport(){
		return (List)getArray("_support", Boolean.class);
	}

	public List<String> getVariableNames(){
		return (List)getListLike("variable_names", String.class);
	}

	public Map<String, String> getVariableDTypes(){
		return (Map)getDict("_variable_dtypes");
	}

	public Map<String, Map<String, ?>> getVariableStats(){
		return (Map)getDict("_variable_stats");
	}
}