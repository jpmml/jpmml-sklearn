/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.ensemble.hist_gradient_boosting;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.TypeInfo;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn.compose.ColumnTransformer;
import sklearn.preprocessing.OrdinalEncoder;

public class HistGradientBoostingUtil {

	private HistGradientBoostingUtil(){
	}

	static
	public Schema preprocess(ColumnTransformer preprocessor, Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		Label label = schema.getLabel();
		List<Feature> features = new ArrayList<>(schema.getFeatures());

		ColumnTransformer filterPreprocessor = new ColumnTransformer(preprocessor.getPythonModule(), preprocessor.getPythonName()){

			{
				putAll(preprocessor);
			}

			@Override
			public List<Object[]> getFittedTransformers(){
				List<Object[]> fittedTransformers = super.getFittedTransformers();

				ClassDictUtil.checkSize(2, fittedTransformers);

				List<Object[]> result = new AbstractList<Object[]>(){

					@Override
					public int size(){
						return fittedTransformers.size();
					}

					@Override
					public Object[] get(int index){
						Object[] fittedTransformer = fittedTransformers.get(index);

						Transformer transformer = ColumnTransformer.getTransformer(fittedTransformer);

						if(transformer instanceof OrdinalEncoder){
							OrdinalEncoder ordinalEncoder = (OrdinalEncoder)transformer;

							List<Feature> rowFeatures = ColumnTransformer.getFeatures(fittedTransformer, features, encoder);

							OrdinalEncoder filterOrdinalEncoder = new OrdinalEncoder(ordinalEncoder.getPythonModule(), ordinalEncoder.getPythonName()){

								{
									putAll(ordinalEncoder);
								}

								@Override
								public List<List<?>> getCategories(){
									List<List<?>> categories = super.getCategories();

									ClassDictUtil.checkSize(categories, rowFeatures);

									List<List<?>> result = new AbstractList<List<?>>(){

										@Override
										public int size(){
											return categories.size();
										}

										@Override
										public List<?> get(int index){
											Feature rowFeature = rowFeatures.get(index);

											if(rowFeature instanceof CategoricalFeature){
												CategoricalFeature categoricalFeature = (CategoricalFeature)rowFeature;

												return categoricalFeature.getValues();
											}

											return categories.get(index);
										}
									};

									return result;
								}

								@Override
								public TypeInfo getDType(){
									TypeInfo result = new TypeInfo(){

										@Override
										public DataType getDataType(){
											return DataType.INTEGER;
										}
									};

									return result;
								}
							};

							ColumnTransformer.setTransformer(fittedTransformer, filterOrdinalEncoder);
						}

						return fittedTransformer;
					}
				};

				return result;
			}
		};

		List<Feature> filterFeatures = filterPreprocessor.encode(features, encoder);

		return new Schema(encoder, label, filterFeatures);
	}

	static
	public MiningModel encodeHistGradientBoosting(List<List<TreePredictor>> predictors, BinMapper binMapper, List<? extends Number> baselinePredictions, int column, Schema schema){
		List<TreePredictor> treePredictors = predictors.stream()
			.map(predictor -> predictor.get(column))
			.collect(Collectors.toList());

		Number baselinePrediction = baselinePredictions.get(column);

		return encodeHistGradientBoosting(treePredictors, binMapper, baselinePrediction, schema);
	}

	static
	public MiningModel encodeHistGradientBoosting(List<TreePredictor> treePredictors, BinMapper binMapper, Number baselinePrediction, Schema schema){
		ContinuousLabel continuousLabel = (ContinuousLabel)schema.getLabel();

		PredicateManager predicateManager = new PredicateManager();

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		List<TreeModel> treeModels = new ArrayList<>();

		for(TreePredictor treePredictor : treePredictors){
			TreeModel treeModel = TreePredictorUtil.encodeTreeModel(treePredictor, binMapper, predicateManager, segmentSchema);

			treeModels.add(treeModel);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(continuousLabel))
			.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.SUM, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels))
			.setTargets(ModelUtil.createRescaleTargets(null, baselinePrediction, continuousLabel));

		return miningModel;
	}
}