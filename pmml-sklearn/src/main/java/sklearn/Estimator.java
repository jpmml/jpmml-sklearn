/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import numpy.core.NDArrayUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.DiscreteLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn2pmml.Customization;
import sklearn2pmml.HasPMMLOptions;
import sklearn2pmml.HasPMMLSegmentId;
import sklearn2pmml.SkLearn2PMMLFields;

abstract
public class Estimator extends Step implements HasNumberOfOutputs, HasPMMLOptions<Estimator>, HasPMMLSegmentId<Estimator> {

	public Estimator(String module, String name){
		super(module, name);
	}

	abstract
	public MiningFunction getMiningFunction();

	abstract
	public boolean isSupervised();

	abstract
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder);

	abstract
	public Model encodeModel(Schema schema);

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public DataType getDataType(){
		return DataType.DOUBLE;
	}

	@Override
	public int getNumberOfFeatures(){

		// SkLearn 1.0+
		nFeaturesIn:
		if(containsKey(SkLearnFields.N_FEATURES_IN)){

			// Deprecated attributes are explicitly set to None values
			if(get(SkLearnFields.N_FEATURES_IN) == null){
				break nFeaturesIn;
			}

			return getInteger(SkLearnFields.N_FEATURES_IN);
		} // End if

		// SkLearn 0.24
		if(containsKey(SkLearnFields.N_FEATURES)){
			return getInteger(SkLearnFields.N_FEATURES);
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public int getNumberOfOutputs(){

		if(containsKey(SkLearnFields.N_OUTPUTS)){
			return getInteger(SkLearnFields.N_OUTPUTS);
		}

		return HasNumberOfOutputs.UNKNOWN;
	}

	public String getAlgorithmName(){
		return getClassName();
	}

	public Model encode(Schema schema){
		checkVersion();

		checkLabel(schema.getLabel());
		checkFeatures(schema.getFeatures());

		schema = configureSchema(schema);

		Model model = encodeModel(schema);

		String modelName = model.getModelName();
		if(modelName == null){
			String pmmlName = getPMMLName();

			if(pmmlName != null){
				model.setModelName(pmmlName);
			}
		}

		String algorithmName = model.getAlgorithmName();
		if(algorithmName == null){
			String pyClassName = getAlgorithmName();

			model.setAlgorithmName(pyClassName);
		}

		addFeatureImportances(model, schema);

		return model;
	}

	public Model encode(Object segmentId, Schema schema){
		Object prevSegmentId = getPMMLSegmentId();

		try {
			setPMMLSegmentId(segmentId);

			return encode(schema);
		} finally {
			setPMMLSegmentId(prevSegmentId);
		}
	}

	public void checkLabel(Label label){
		boolean supervised = isSupervised();

		if(supervised){

			if(label == null){
				throw new IllegalArgumentException("Expected a label, got no label");
			}
		} else

		{
			if(label != null){
				throw new IllegalArgumentException("Expected no label, got " + label);
			}
		}
	}

	public void checkFeatures(List<? extends Feature> features){
		StepUtil.checkNumberOfFeatures(this, features);
	}

	public Schema configureSchema(Schema schema){
		return schema;
	}

	public void addFeatureImportances(Model model, Schema schema){
		List<? extends Number> featureImportances = getPMMLFeatureImportances();
		if(featureImportances == null){
			featureImportances = getFeatureImportances();
		}

		ModelEncoder encoder = (ModelEncoder)schema.getEncoder();
		List<? extends Feature> features = schema.getFeatures();

		if(featureImportances != null){
			ClassDictUtil.checkSize(features, featureImportances);

			for(int i = 0; i < features.size(); i++){
				Feature feature = features.get(i);
				Number featureImportance = featureImportances.get(i);

				encoder.addFeatureImportance(model, feature, featureImportance);
			}
		}
	}

	public Object getOption(String key, Object defaultValue){
		Map<String, ?> pmmlOptions = getPMMLOptions();

		if(pmmlOptions != null && pmmlOptions.containsKey(key)){
			return pmmlOptions.get(key);
		} // End if

		// XXX
		if(containsKey(key)){
			logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, SkLearn2PMMLFields.PMML_OPTIONS) + "\' is not set. Falling back to the surrogate attribute \'" + ClassDictUtil.formatMember(this, key) + "\'");

			return get(key);
		}

		return defaultValue;
	}

	public void putOption(String key, Object value){
		putOptions(Collections.singletonMap(key, value));
	}

	@SuppressWarnings("unchecked")
	public void putOptions(Map<String, ?> options){
		Map<String, Object> pmmlOptions = (Map<String, Object>)getPMMLOptions();

		if(pmmlOptions == null){
			pmmlOptions = new LinkedHashMap<>();

			setPMMLOptions(pmmlOptions);
		}

		pmmlOptions.putAll(options);
	}

	public List<? extends Customization> getPMMLCustomizations(){

		if(!containsKey(SkLearn2PMMLFields.PMML_CUSTOMIZATIONS)){
			return null;
		}

		return getArray(SkLearn2PMMLFields.PMML_CUSTOMIZATIONS, Customization.class);
	}

	public Estimator setPMMLCustomizations(List<? extends Customization> pmmlCustomizations){
		put(SkLearn2PMMLFields.PMML_CUSTOMIZATIONS, NDArrayUtil.toArray(pmmlCustomizations));

		return this;
	}

	public boolean hasFeatureImportances(){
		return containsKey(SkLearnFields.FEATURE_IMPORTANCES) || containsKey(SkLearn2PMMLFields.PMML_FEATURE_IMPORTANCES);
	}

	public List<? extends Number> getFeatureImportances(){

		if(!containsKey(SkLearnFields.FEATURE_IMPORTANCES)){
			return null;
		}

		return getNumberArray(SkLearnFields.FEATURE_IMPORTANCES);
	}

	public List<? extends Number> getPMMLFeatureImportances(){

		if(!containsKey(SkLearn2PMMLFields.PMML_FEATURE_IMPORTANCES)){
			return null;
		}

		return getNumberArray(SkLearn2PMMLFields.PMML_FEATURE_IMPORTANCES);
	}

	public Estimator setPMMLFeatureImportances(List<? extends Number> pmmlFeatureImportances){
		put(SkLearn2PMMLFields.PMML_FEATURE_IMPORTANCES, NDArrayUtil.toArray(pmmlFeatureImportances));

		return this;
	}

	@Override
	public Map<String, ?> getPMMLOptions(){
		Object value = get(SkLearn2PMMLFields.PMML_OPTIONS);

		if(value == null){
			return null;
		}

		return getDict(SkLearn2PMMLFields.PMML_OPTIONS);
	}

	@Override
	public Estimator setPMMLOptions(Map<String, ?> pmmlOptions){
		put(SkLearn2PMMLFields.PMML_OPTIONS, pmmlOptions);

		return this;
	}

	@Override
	public Object getPMMLSegmentId(){
		return getOptionalScalar(SkLearn2PMMLFields.PMML_SEGMENT_ID);
	}

	@Override
	public Estimator setPMMLSegmentId(Object pmmlSegmentId){

		if(pmmlSegmentId != null){
			put(SkLearn2PMMLFields.PMML_SEGMENT_ID, pmmlSegmentId);
		} else

		{
			remove(SkLearn2PMMLFields.PMML_SEGMENT_ID);
		}

		return this;
	}

	public List<OutputField> createPredictProbaFields(DataType dataType, DiscreteLabel discreteLabel){
		Object pmmlSegmentId = getPMMLSegmentId();

		if(this instanceof HasClasses){
			HasClasses hasClasses = (HasClasses)this;
		} else

		{
			throw new IllegalArgumentException();
		}

		List<?> values = discreteLabel.getValues();

		return values.stream()
			.map(value -> {
				String name;

				if(pmmlSegmentId != null){
					name = FieldNameUtil.create(Classifier.FIELD_PROBABILITY, pmmlSegmentId, value);
				} else

				{
					name = FieldNameUtil.create(Classifier.FIELD_PROBABILITY, value);
				}

				return ModelUtil.createProbabilityField(name, dataType, value);
			})
			.collect(Collectors.toList());
	}

	public OutputField createApplyField(DataType dataType){
		Object pmmlSegmentId = getPMMLSegmentId();

		String name;

		if(this instanceof HasApplyField){
			HasApplyField hasApplyField = (HasApplyField)this;

			name = hasApplyField.getApplyField();
		} else

		{
			throw new IllegalArgumentException();
		} // End if

		if(pmmlSegmentId != null){
			name = FieldNameUtil.create(name, pmmlSegmentId);
		}

		return ModelUtil.createEntityIdField(name, dataType);
	}

	public OutputField encodeApplyOutput(Model model, DataType dataType){
		OutputField applyField = createApplyField(dataType);

		Output output = ModelUtil.ensureOutput(model);

		(output.getOutputFields()).add(applyField);

		return applyField;
	}

	public OutputField createMultiApplyField(DataType dataType, String segmentId){
		Object pmmlSegmentId = getPMMLSegmentId();

		String name;

		if(this instanceof HasMultiApplyField){
			HasMultiApplyField hasMultiApplyField = (HasMultiApplyField)this;

			name = hasMultiApplyField.getMultiApplyField(segmentId);
		} else

		{
			throw new IllegalArgumentException();
		} // End if

		if(pmmlSegmentId != null){
			name = FieldNameUtil.create(name, pmmlSegmentId);
		}

		OutputField result = ModelUtil.createEntityIdField(name, dataType);

		result.setSegmentId(segmentId);

		return result;
	}

	public List<OutputField> encodeMultiApplyOutput(Model model, DataType dataType, List<String> segmentIds){
		List<OutputField> applyFields = new ArrayList<>();

		for(String segmentId : segmentIds){
			OutputField applyField = createMultiApplyField(dataType, segmentId);

			applyFields.add(applyField);
		}

		Output output = ModelUtil.ensureOutput(model);

		(output.getOutputFields()).addAll(applyFields);

		return applyFields;
	}

	static
	protected String extractArguments(String function, String name){

		if(name.startsWith(function + "(") && name.endsWith(")")){
			return name.substring((function + "(").length(), name.length() - ")".length());
		}

		return name;
	}

	public static final String FIELD_APPLY = "apply";
	public static final String FIELD_DECISION_FUNCTION = "decisionFunction";
	public static final String FIELD_PREDICT = "predict";

	private static final Logger logger = LoggerFactory.getLogger(Estimator.class);
}