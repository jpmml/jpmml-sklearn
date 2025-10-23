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
package interpret.glassbox.ebm;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import numpy.core.ScalarUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.Interval;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.NamespacePrefixes;
import org.dmg.pmml.OpType;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.XMLUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import sklearn.Estimator;

public class ExplainableBoostingUtil {

	private ExplainableBoostingUtil(){
	}

	static
	public <E extends Estimator & HasExplainableBooster> List<Feature> encodeExplainableBooster(E estimator, Schema schema){
		List<List<?>> bins = estimator.getBins();
		List<String> featureTypesIn = estimator.getFeatureTypesIn();
		List<Object[]> termFeatures = estimator.getTermFeatures();
		List<HasArray> termScores = estimator.getTermScores();

		ClassDictUtil.checkSize(bins, featureTypesIn);
		ClassDictUtil.checkSize(termFeatures, termScores);

		ModelEncoder encoder = schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<List<CategoricalFeature>> binLevelFeatures = new ArrayList<>();

		for(int i = 0; i < bins.size(); i++){
			Feature feature = features.get(i);
			List<?> binLevels = bins.get(i);
			String featureType = featureTypesIn.get(i);

			binLevelFeatures.add(encodeBinLevelFeatures(feature, binLevels, featureType, encoder));
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < termFeatures.size(); i++){
			Object[] termFeature = termFeatures.get(i);
			HasArray termScore = termScores.get(i);

			Feature feature = encodeLookupFeature(termFeature, termScore, binLevelFeatures, encoder);

			result.add(feature);
		}

		return result;
	}

	static
	private List<CategoricalFeature> encodeBinLevelFeatures(Feature feature, List<?> binLevels, String featureType, ModelEncoder encoder){
		List<CategoricalFeature> result = new ArrayList<>();

		for(int i = 0; i < binLevels.size(); i++){
			Object binLevel = binLevels.get(i);

			switch(featureType){
				case HasExplainableBooster.FEATURETYPE_CONTINUOUS:
					{
						result.add(binContinuous(feature, (HasArray)binLevel, (binLevels.size() > 1 ? i : null), encoder));
					}
					break;
				case HasExplainableBooster.FEATURETYPE_NOMINAL:
					{
						result.add(binNominal(feature, (Map<?, ?>)binLevel, encoder));
					}
					break;
				default:
					throw new IllegalArgumentException(featureType);
			}
		}

		return result;
	}

	static
	private IndexFeature binContinuous(Feature feature, HasArray binLevel, Integer binLevelIndex, ModelEncoder encoder){
		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		Discretize discretize = new Discretize(continuousFeature.getName());

		List<Number> bins = (List<Number>)binLevel.getArrayContent();
		if(bins.isEmpty()){
			throw new IllegalArgumentException();
		}

		List<Integer> labelCategories = new ArrayList<>();

		for(int j = 0; j <= bins.size(); j++){
			Number leftMargin = null;
			Number rightMargin = null;

			if(j == 0){
				rightMargin = bins.get(j);
			} else

			if(j == bins.size()){
				leftMargin = bins.get(j - 1);
			} else

			{
				leftMargin = bins.get(j - 1);
				rightMargin = bins.get(j);
			}

			Integer label = j;

			labelCategories.add(label);

			Interval interval = new Interval(Interval.Closure.CLOSED_OPEN, leftMargin, rightMargin);

			DiscretizeBin discretizeBin = new DiscretizeBin(label, interval);

			discretize.addDiscretizeBins(discretizeBin);
		}

		String name;

		if(binLevelIndex != null){
			name = FieldNameUtil.create("bin", continuousFeature, binLevelIndex);
		} else

		{
			name = FieldNameUtil.create("bin", continuousFeature);
		}

		DerivedField derivedField = encoder.createDerivedField(name, OpType.CATEGORICAL, DataType.INTEGER, discretize);

		return new IndexFeature(encoder, derivedField, labelCategories);
	}

	static
	private CategoricalFeature binNominal(Feature feature, Map<?, ?> binLevel, PMMLEncoder encoder){
		List<Map.Entry<?, ?>> entries = (binLevel.entrySet()).stream()
			.sorted((left, right) -> {
				return ((Comparable)left.getValue()).compareTo((right.getValue()));
			})
			.collect(Collectors.toList());

		List<Object> categories = new ArrayList<>();

		for(int j = 0; j < entries.size(); j++){
			Map.Entry<?, ?> entry = entries.get(j);

			Object category = ScalarUtil.decode(entry.getKey());
			Number bin = (Number)entry.getValue();

			if(bin.intValue() != (j + 1)){
				throw new IllegalArgumentException();
			}

			categories.add(category);
		}

		Field<?> field = encoder.toCategorical(feature.getName(), categories);

		// XXX
		field.setDataType(TypeUtil.getDataType(categories, DataType.STRING));

		return new CategoricalFeature(encoder, field, categories);
	}

	static
	private Feature encodeLookupFeature(Object[] termFeature, HasArray termScore, List<List<CategoricalFeature>> binLevelFeatures, PMMLEncoder encoder){
		int[] termScoreShape = termScore.getArrayShape();
		List<Number> termScoreContent = (List<Number>)termScore.getArrayContent();

		int rows;
		int columns;

		if(termFeature.length == 1){
			rows = (termScoreShape[0] - 2);
			columns = 1;
		} else

		if(termFeature.length == 2){
			rows = (termScoreShape[0] - 2);
			columns = (termScoreShape[1] - 2);
		} else

		{
			throw new IllegalArgumentException();
		}

		List<Feature> inputFeatures = new ArrayList<>();

		String outputColumn = NamespacePrefixes.JPMML_INLINETABLE + ":" + "output";

		MapValues mapValues = new MapValues()
			.setMapMissingTo(0d)
			.setOutputColumn(outputColumn);

		Map<String, List<Object>> data = new LinkedHashMap<>();

		for(int j = 0; j < termFeature.length; j++){
			Integer featureIndex = (Integer)termFeature[j];

			List<CategoricalFeature> binnedFeatures = binLevelFeatures.get(featureIndex);

			CategoricalFeature binnedFeature;

			String inputColumn;
			List<Object> categoryValues;

			if(termFeature.length == 1){
				binnedFeature = binnedFeatures.get(0);

				inputColumn = NamespacePrefixes.JPMML_INLINETABLE + ":" + "input";

				categoryValues = (List)binnedFeature.getValues();
			} else

			{
				binnedFeature = binnedFeatures.get(Math.min(binnedFeatures.size() - 1, 1));

				inputColumn = NamespacePrefixes.JPMML_INLINETABLE + ":" + XMLUtil.createTagName("input_" + j);

				categoryValues = new ArrayList<>();

				for(int k = 0, max = (rows * columns); k < max; k++){
					int index;

					if(j == 0){
						index = (k / columns);
					} else

					if(j == 1){
						index = (k % columns);
					} else

					{
						throw new IllegalArgumentException();
					}

					categoryValues.add(binnedFeature.getValue(index));
				}
			}

			inputFeatures.add(binnedFeature);

			FieldColumnPair fieldColumnPair = new FieldColumnPair(binnedFeature.getName(), inputColumn);

			mapValues.addFieldColumnPairs(fieldColumnPair);

			data.put(inputColumn, categoryValues);
		}

		List<Object> outputValues;

		if(termFeature.length == 1){
			outputValues = (List)termScoreContent.subList(1, termScoreContent.size() - 1);
		} else

		if(termFeature.length == 2){
			outputValues = new ArrayList<>();

			for(int row = 0; row < rows; row++){

				for(int column = 0; column < columns; column++){
					Number value = termScoreContent.get((row + 1) * (columns + 2) + (column + 1));

					outputValues.add(value);
				}
			}
		} else

		{
			throw new IllegalArgumentException();
		}

		data.put(outputColumn, outputValues);

		InlineTable inlineTable = PMMLUtil.createInlineTable(data);

		mapValues.setInlineTable(inlineTable);

		String name = FieldNameUtil.create("lookup", inputFeatures);

		DerivedField derivedField = encoder.createDerivedField(name, OpType.CATEGORICAL, DataType.DOUBLE, mapValues);

		Feature feature = new ContinuousFeature(encoder, derivedField);

		return feature;
	}
}