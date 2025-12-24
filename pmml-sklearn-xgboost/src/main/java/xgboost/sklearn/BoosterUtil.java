/*
 * Copyright (c) 2016 Villu Ruusmann
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
package xgboost.sklearn;

import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.dmg.pmml.PMML;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnException;
import org.jpmml.xgboost.ByteOrderUtil;
import org.jpmml.xgboost.FeatureMap;
import org.jpmml.xgboost.GBTree;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.ObjFunction;
import pandas.core.BlockManager;
import pandas.core.DataFrame;
import pandas.core.Index;
import sklearn.Estimator;
import sklearn.preprocessing.OrdinalFeature;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> int getNumberOfFeatures(E estimator){
		Learner learner = getLearner(estimator);

		return learner.num_feature();
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> ObjFunction getObjFunction(E estimator){
		Learner learner = getLearner(estimator);

		return learner.obj();
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> MiningModel encodeModel(E estimator, Schema schema){
		Booster booster = estimator.getBooster();

		Learner learner = getLearner(estimator);

		Map<String, ?> options = getOptions(booster, learner, estimator);

		Integer ntreeLimit = (Integer)options.get(HasXGBoostOptions.OPTION_NTREE_LIMIT);

		MiningModel miningModel = learner.encodeModel(ntreeLimit, schema);

		return miningModel;
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> Schema configureSchema(E estimator, Schema schema){
		Booster booster = estimator.getBooster();
		Object missing = estimator.getOptionalObject("missing");

		Learner learner = getLearner(estimator);

		Map<String, ?> options = getOptions(booster, learner, estimator);

		Function<Feature, Feature> function = new Function<Feature, Feature>(){

			@Override
			public Feature apply(Feature feature){

				if(feature instanceof OrdinalFeature){
					OrdinalFeature ordinalFeature = (OrdinalFeature)feature;

					List<?> categories = ordinalFeature.getValues();
					if(!categories.isEmpty()){
						Object lastCategory = categories.get(categories.size() - 1);

						if(ValueUtil.isNaN(missing) && ValueUtil.isNaN(lastCategory)){
							feature = new CategoricalFeature(ordinalFeature.getEncoder(), ordinalFeature, categories.subList(0, categories.size() - 1)){

								@Override
								public ContinuousFeature toContinuousFeature(){
									throw new UnsupportedOperationException();
								}
							};
						}
					}
				}

				return feature;
			}
		};

		schema = schema.toTransformedSchema(function);

		Schema xgbSchema = learner.toXGBoostSchema(schema);

		xgbSchema = learner.configureSchema(options, xgbSchema);

		return xgbSchema;
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> MiningModel configureModel(E estimator, MiningModel miningModel){
		Booster booster = estimator.getBooster();

		Learner learner = getLearner(estimator);

		Map<String, ?> options = getOptions(booster, learner, estimator);

		miningModel = learner.configureModel(options, miningModel);

		return miningModel;
	}

	static
	public PMML encodePMML(Booster booster){
		FeatureMap featureMap = null;

		DataFrame fmap = booster.getFMap();
		if(fmap != null){
			featureMap = parseFMap(fmap);
		}

		Learner learner = booster.getLearner(ByteOrder.nativeOrder(), null);

		if(featureMap == null){
			FeatureMap embeddedFeatureMap = learner.encodeFeatureMap();

			if(embeddedFeatureMap == null || embeddedFeatureMap.isEmpty()){
				String message = "The booster object does not specify feature information";
				String solution = "Set the \'" + ClassDictUtil.formatMember(booster, "fmap") + "\' attribute, or re-train the booster with a DMatrix that has both feature names and feature types set";

				throw new SkLearnException(message)
					.setSolution(solution);
			}
		}

		Map<String, ?> options = getOptions(booster, learner);

		return learner.encodePMML(options, null, null, featureMap);
	}

	static
	public <E extends Estimator & HasBooster & HasXGBoostOptions> PMML encodePMML(E estimator){
		Booster booster = estimator.getBooster();

		Learner learner = getLearner(estimator);

		Map<String, ?> options = getOptions(booster, learner, estimator);

		return learner.encodePMML(options, null, null, null);
	}

	static
	private <E extends Estimator & HasBooster & HasXGBoostOptions> Learner getLearner(E estimator){
		Booster booster = estimator.getBooster();

		String byteOrder = (String)estimator.getOption(HasXGBoostOptions.OPTION_BYTE_ORDER, (ByteOrder.nativeOrder()).toString());
		String charset = (String)estimator.getOption(HasXGBoostOptions.OPTION_CHARSET, null);

		return booster.getLearner(ByteOrderUtil.forValue(byteOrder), charset);
	}

	static
	private Map<String, ?> getOptions(Booster booster, Learner learner){
		Map<String, Object> result = new LinkedHashMap<>();

		Integer bestNTreeLimit = booster.getBestNTreeLimit();

		if(bestNTreeLimit == null){
			Integer bestIteration = learner.getBestIteration();

			if(bestIteration != null){
				bestNTreeLimit = (bestIteration + 1);
			}
		}

		result.put(HasXGBoostOptions.OPTION_NTREE_LIMIT, bestNTreeLimit);

		return result;
	}

	static
	private <E extends Estimator & HasBooster & HasXGBoostOptions> Map<String, ?> getOptions(Booster booster, Learner learner, E estimator){
		GBTree gbtree = learner.gbtree();

		Map<String, Object> result = new LinkedHashMap<>();

		Integer bestNTreeLimit = booster.getBestNTreeLimit();

		// XGBoost 1.7
		if(bestNTreeLimit == null){
			bestNTreeLimit = (Integer)estimator.getOptionalScalar("best_ntree_limit");
		} // End if

		// XGBoost 2.0+
		if(bestNTreeLimit == null){
			Integer bestIteration = learner.getBestIteration();

			if(bestIteration != null){
				bestNTreeLimit = (bestIteration + 1);
			}
		}

		Integer ntreeLimit = (Integer)estimator.getOption(HasXGBoostOptions.OPTION_NTREE_LIMIT, bestNTreeLimit);
		result.put(HasXGBoostOptions.OPTION_NTREE_LIMIT, ntreeLimit);

		Number missing = (Number)estimator.getOptionalScalar("missing");
		result.put(HasXGBoostOptions.OPTION_MISSING, missing);

		Boolean compact = (Boolean)estimator.getOption(HasXGBoostOptions.OPTION_COMPACT, !gbtree.hasCategoricalSplits());
		Boolean inputFloat = (Boolean)estimator.getOption(HasXGBoostOptions.OPTION_INPUT_FLOAT, null);
		Boolean numeric = (Boolean)estimator.getOption(HasXGBoostOptions.OPTION_NUMERIC, Boolean.TRUE);
		Boolean prune = (Boolean)estimator.getOption(HasXGBoostOptions.OPTION_PRUNE, Boolean.TRUE);

		result.put(HasXGBoostOptions.OPTION_COMPACT, compact);
		result.put(HasXGBoostOptions.OPTION_INPUT_FLOAT, inputFloat);
		result.put(HasXGBoostOptions.OPTION_NUMERIC, numeric);
		result.put(HasXGBoostOptions.OPTION_PRUNE, prune);

		return result;
	}

	static
	private FeatureMap parseFMap(DataFrame fmap){
		BlockManager data = fmap.getData();

		Index columnAxis = data.getColumnAxis();
		Index rowAxis = data.getRowAxis();

		if(!(Arrays.asList("id", "name", "type")).equals(columnAxis.getValues())){
			throw new IllegalArgumentException();
		}

		List<HasArray> blockValues = data.getBlockValues();

		HasArray idColumn = blockValues.get(0);
		HasArray nameTypeColumns = blockValues.get(1);

		List<?> nameTypeContent = nameTypeColumns.getArrayContent();
		int[] nameTypeShape = nameTypeColumns.getArrayShape();

		List<?> nameValues = CMatrixUtil.getRow(nameTypeContent, nameTypeShape[0], nameTypeShape[1], 0);
		List<?> typeValues = CMatrixUtil.getRow(nameTypeContent, nameTypeShape[0], nameTypeShape[1], 1);

		FeatureMap result = new FeatureMap();

		for(int i = 0; i < nameTypeShape[1]; i++){
			String name = (String)nameValues.get(i);
			String type = (String)typeValues.get(i);

			result.addEntry(name, type);
		}

		return result;
	}
}