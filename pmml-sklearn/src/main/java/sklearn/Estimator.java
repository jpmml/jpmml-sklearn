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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import numpy.core.NDArrayUtil;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn2pmml.SkLearn2PMMLFields;

abstract
public class Estimator extends Step implements HasNumberOfOutputs {

	public Estimator(String module, String name){
		super(module, name);
	}

	abstract
	public MiningFunction getMiningFunction();

	abstract
	public Model encodeModel(Schema schema);

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

	@Override
	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	@Override
	public DataType getDataType(){
		return DataType.DOUBLE;
	}

	public boolean isSupervised(){
		MiningFunction miningFunction = getMiningFunction();

		switch(miningFunction){
			case CLASSIFICATION:
			case REGRESSION:
				return true;
			case CLUSTERING:
				return false;
			default:
				throw new IllegalArgumentException();
		}
	}

	public String getAlgorithmName(){
		return getClassName();
	}

	public Model encode(Schema schema){
		checkLabel(schema.getLabel());
		checkFeatures(schema.getFeatures());

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

	public Map<String, ?> getPMMLOptions(){
		Object value = get(SkLearn2PMMLFields.PMML_OPTIONS);

		if(value == null){
			return null;
		}

		return getDict(SkLearn2PMMLFields.PMML_OPTIONS);
	}

	public Estimator setPMMLOptions(Map<String, ?> pmmlOptions){
		put(SkLearn2PMMLFields.PMML_OPTIONS, pmmlOptions);

		return this;
	}

	public String getSkLearnVersion(){
		return getOptionalString(SkLearnFields.SKLEARN_VERSION);
	}

	public static final String FIELD_APPLY = "apply";
	public static final String FIELD_DECISION_FUNCTION = "decisionFunction";
	public static final String FIELD_PREDICT = "predict";

	private static final Logger logger = LoggerFactory.getLogger(Estimator.class);
}