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
package sklego.meta;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import numpy.core.ScalarUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.DiscreteLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.OrdinalLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Classifier;
import sklearn.EstimatorException;
import sklearn.HasClasses;
import sklearn.SkLearnFields;
import sklearn2pmml.SkLearn2PMMLFields;

public class OrdinalClassifier extends Classifier {

	public OrdinalClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		Map<?, ? extends Classifier> estimators = getEstimators();
		Map<?, ?> estimatorCategories = getEstimatorCategories();

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		OrdinalLabel ordinalLabel = (OrdinalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		SchemaUtil.checkSize(estimators.size() + 1, ordinalLabel);

		List<Model> models = new ArrayList<>();

		List<Feature> probabilityFeatures = new ArrayList<>();

		for(int i = 0, max = (ordinalLabel.size() - 1); i < max; i++){
			Object category = ordinalLabel.getValue(i);

			Classifier estimator;

			if(estimatorCategories != null && !estimatorCategories.isEmpty()){
				Object estimatorCategory = estimatorCategories.get(category);

				if(estimatorCategory == null){
					throw new IllegalArgumentException();
				}

				estimator = estimators.get(estimatorCategory);
			} else

			{
				estimator = estimators.get(category);
			} // End if

			if(estimator == null){
				throw new IllegalArgumentException();
			} // End if

			if(!estimator.hasProbabilityDistribution()){
				throw new EstimatorException(estimator, HasClasses.class);
			}

			CategoricalLabel segmentLabel = new CategoricalLabel(DataType.DOUBLE, Arrays.asList("<=" + ValueUtil.asString(category), ">" + ValueUtil.asString(category)));

			Schema segmentSchema = schema.toRelabeledSchema(segmentLabel);

			Model model = estimator.encode(segmentSchema);

			// XXX
			String name = FieldNameUtil.create(Classifier.FIELD_PROBABILITY, segmentLabel.getValue(1));

			List<Feature> segmentFeatures = encoder.export(model, name);
			if(segmentFeatures.size() != 1){
				throw new IllegalArgumentException();
			}

			models.add(model);

			probabilityFeatures.addAll(segmentFeatures);
		}

		SchemaUtil.checkSize(estimators.size(), probabilityFeatures);

		List<RegressionTable> regressionTables = new ArrayList<>();

		// The first category and one or more intermediate categories
		for(int i = 0; i < estimators.size(); i++){
			RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(probabilityFeatures.get(i)), Collections.singletonList(1), 0d)
				.setTargetCategory(ordinalLabel.getValue(i));

			regressionTables.add(regressionTable);
		}

		// The final category
		{
			RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.emptyList(), Collections.emptyList(), 1d)
				.setTargetCategory(ordinalLabel.getValue(estimators.size()));

			regressionTables.add(regressionTable);
		}

		RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(ordinalLabel), regressionTables)
			.setNormalizationMethod(RegressionModel.NormalizationMethod.NONE);

		encodePredictProbaOutput(regressionModel, DataType.DOUBLE, ordinalLabel);

		models.add(regressionModel);

		return MiningModelUtil.createModelChain(models, Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	@Override
	protected DiscreteLabel encodeLabel(String name, OpType opType, DataType dataType, List<?> categories, SkLearnEncoder encoder){
		return super.encodeLabel(name, OpType.ORDINAL, DataType.STRING, categories, encoder);
	}

	public Classifier getEstimator(){
		return getClassifier("estimator");
	}

	public Map<?, ? extends Classifier> getEstimators(){
		Map<?, ?> estimators = getDict("estimators_");

		Function<Object, Object> keyFunction = new Function<Object, Object>(){

			@Override
			public Object apply(Object object){
				object = ScalarUtil.decode(object);

				return Classifier.canonicalizeValue(object);
			}
		};

		Function<Object, Classifier> valueFunction = new CastFunction<Classifier>(Classifier.class){

			@Override
			protected String formatMessage(Object object){
				return "The item value object (" + ClassDictUtil.formatClass(object) + ") is not a supported Classifier";
			}
		};

		Map<?, ? extends Classifier> result = (estimators.entrySet()).stream()
			.collect(Collectors.toMap(entry -> keyFunction.apply(entry.getKey()), entry -> valueFunction.apply(entry.getValue())));

		return result;
	}

	private Map<?, ?> getEstimatorCategories(){

		if(!hasattr(SkLearn2PMMLFields.PMML_CLASSES)){
			return null;
		}

		List<?> classes = getClasses(SkLearnFields.CLASSES);
		List<?> pmmlClasses = getClasses(SkLearn2PMMLFields.PMML_CLASSES);

		ClassDictUtil.checkSize(classes, pmmlClasses);

		Map<Object, Object> result = new LinkedHashMap<>();

		for(int i = 0; i < classes.size(); i++){
			result.put(pmmlClasses.get(i), classes.get(i));
		}

		return result;
	}
}