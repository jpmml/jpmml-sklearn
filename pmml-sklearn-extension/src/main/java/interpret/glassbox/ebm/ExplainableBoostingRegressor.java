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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.Interval;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.Model;
import org.dmg.pmml.NamespacePrefixes;
import org.dmg.pmml.OpType;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.Label;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.XMLUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import sklearn.Regressor;

public class ExplainableBoostingRegressor extends Regressor {

	public ExplainableBoostingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<List<HasArray>> bins = getBins();
		Number intercept = getIntercept();
		RegressionModel.NormalizationMethod link = parseLink(getLink());
		List<Object[]> termFeatures = getTermFeatures();
		List<HasArray> termScores = getTermScores();

		ClassDictUtil.checkSize(termFeatures, termScores);

		PMMLEncoder encoder = schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<List<IndexFeature>> binLevelFeatures = new ArrayList<>();

		for(int i = 0; i < bins.size(); i++){
			Feature feature = features.get(i);
			List<HasArray> binLevels = bins.get(i);

			binLevelFeatures.add(encodeBinLevelFeatures(feature, binLevels, encoder));
		}

		List<Feature> lookupFeatures = new ArrayList<>();
		List<Number> coefficients = new ArrayList<>();

		for(int i = 0; i < termFeatures.size(); i++){
			Object[] termFeature = termFeatures.get(i);
			HasArray termScore = termScores.get(i);

			Feature feature = encodeLookupFeature(termFeature, termScore, binLevelFeatures, encoder);

			lookupFeatures.add(feature);
			coefficients.add(1d);
		}

		return RegressionModelUtil.createRegression(lookupFeatures, coefficients, intercept, link, schema);
	}

	static
	private List<IndexFeature> encodeBinLevelFeatures(Feature feature, List<HasArray> binLevels, PMMLEncoder encoder){
		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		List<IndexFeature> result = new ArrayList<>();

		for(int i = 0; i < binLevels.size(); i++){
			HasArray binLevel = binLevels.get(i);

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

				Interval interval = new Interval(Interval.Closure.CLOSED_OPEN)
					.setLeftMargin(leftMargin)
					.setRightMargin(rightMargin);

				DiscretizeBin discretizeBin = new DiscretizeBin(label, interval);

				discretize.addDiscretizeBins(discretizeBin);
			}

			String name;

			if(binLevels.size() > 1){
				name = FieldNameUtil.create("bin", continuousFeature, i);
			} else

			{
				name = FieldNameUtil.create("bin", continuousFeature);
			}

			DerivedField derivedField = encoder.createDerivedField(name, OpType.CATEGORICAL, DataType.INTEGER, discretize);

			result.add(new IndexFeature(encoder, derivedField, labelCategories));
		}

		return result;
	}

	static
	private Feature encodeLookupFeature(Object[] termFeature, HasArray termScore, List<List<IndexFeature>> binLevelFeatures, PMMLEncoder encoder){
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

		Map<String, List<Number>> data = new LinkedHashMap<>();

		for(int j = 0; j < termFeature.length; j++){
			Integer featureIndex = (Integer)termFeature[j];

			List<IndexFeature> binnedFeatures = binLevelFeatures.get(featureIndex);

			IndexFeature binnedFeature;

			String inputColumn;
			List<Number> categoryValues;

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

					if(j == 0){
						categoryValues.add(k / columns);
					} else

					if(j == 1){
						categoryValues.add(k % columns);
					} else

					{
						throw new IllegalArgumentException();
					}
				}
			}

			inputFeatures.add(binnedFeature);

			FieldColumnPair fieldColumnPair = new FieldColumnPair(binnedFeature.getName(), inputColumn);

			mapValues.addFieldColumnPairs(fieldColumnPair);

			data.put(inputColumn, categoryValues);
		}

		List<Number> outputValues;

		if(termFeature.length == 1){
			outputValues = termScoreContent.subList(1, termScoreContent.size() - 1);
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

	public List<List<HasArray>> getBins(){
		return (List)getList("bins_", List.class);
	}

	public Number getIntercept(){
		return getNumber("intercept_");
	}

	public String getLink(){
		return getEnum("link_", this::getString, Arrays.asList(ExplainableBoostingRegressor.LINK_IDENTITY));
	}

	public List<Object[]> getTermFeatures(){
		return getTupleList("term_features_");
	}

	public List<HasArray> getTermScores(){
		return getArrayList("term_scores_");
	}

	static
	private RegressionModel.NormalizationMethod parseLink(String link){

		switch(link){
			case ExplainableBoostingRegressor.LINK_IDENTITY:
				return RegressionModel.NormalizationMethod.NONE;
			default:
				throw new IllegalArgumentException(link);
		}
	}

	private static final String LINK_IDENTITY = "identity";
}