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

import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

abstract
public class Estimator extends Step implements HasNumberOfFeatures, HasType {

	public Estimator(String module, String name){
		super(module, name);
	}

	abstract
	public MiningFunction getMiningFunction();

	abstract
	public Model encodeModel(Schema schema);

	@Override
	public int getNumberOfFeatures(){

		if(containsKey("n_features_")){
			return getInteger("n_features_");
		}

		return HasNumberOfFeatures.UNKNOWN;
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

	public Model encode(Schema schema){
		List<? extends Number> featureImportances = getFeatureImportances();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Model model = encodeModel(schema);

		if(featureImportances != null){
			ClassDictUtil.checkSize(features, featureImportances);

			for(int i = 0; i < features.size(); i++){
				Feature feature = features.get(i);
				Number featureImportance = featureImportances.get(i);

				SkLearnEncoder encoder = (SkLearnEncoder)feature.getEncoder();

				encoder.addFeatureImportance(model, feature.getName(), featureImportance);
			}
		}

		return model;
	}

	public Object getOption(String key, Object defaultValue){
		Map<String, ?> pmmlOptions = getPMMLOptions();

		if(pmmlOptions != null && pmmlOptions.containsKey(key)){
			return pmmlOptions.get(key);
		} // End if

		// XXX
		if(containsKey(key)){
			logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "pmml_options_") + "\' is not set. Falling back to the surrogate attribute \'" + ClassDictUtil.formatMember(this, key) + "\'");

			return get(key);
		}

		return defaultValue;
	}

	public List<? extends Number> getFeatureImportances(){

		if(!containsKey("feature_importances_")){
			return null;
		}

		return getNumberArray("feature_importances_");
	}

	public Map<String, ?> getPMMLOptions(){
		Object value = get("pmml_options_");

		if(value == null){
			return null;
		}

		return getDict("pmml_options_");
	}

	public Estimator setPMMLOptions(Map<String, ?> pmmlOptions){
		put("pmml_options_", pmmlOptions);

		return this;
	}

	public String getSkLearnVersion(){
		return getOptionalString("_sklearn_version");
	}

	private static final Logger logger = LoggerFactory.getLogger(Estimator.class);
}