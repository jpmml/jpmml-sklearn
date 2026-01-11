/*
 * Copyright (c) 2020 Villu Ruusmann
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

import org.jpmml.converter.ConversionException;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.python.Attribute;
import org.jpmml.python.PythonObject;
import org.jpmml.sklearn.SkLearnException;
import sklearn2pmml.SkLearn2PMMLFields;

abstract
public class Step extends PythonObject implements HasNumberOfFeatures, HasType {

	public Step(String module, String name){
		super(module, name);
	}

	public void checkVersion(){
		checkSkLearnVersion();
	}

	public void checkSkLearnVersion(){
		String sklearnVersion = getSkLearnVersion();
		String supportedSklearnVersion = "1.8.0";

		if(sklearnVersion != null && VersionUtil.compareVersion(sklearnVersion, supportedSklearnVersion) > 0){
			String message = "Scikit-Learn version " + ExceptionUtil.formatVersion(sklearnVersion) + " is not supported";
			String solution = "Upgrade the converter to the latest version, or downgrade Scikit-Learn to version " + ExceptionUtil.formatVersion(supportedSklearnVersion);

			throw new SkLearnException(message)
				.setSolution(solution);
		}
	}

	public List<String> getFeatureNamesIn(){

		if(hasattr(SkLearnFields.FEATURE_NAMES_IN)){
			return getStringListLike(SkLearnFields.FEATURE_NAMES_IN);
		}

		return null;
	}

	public Map<String, ?> getInputTags(){
		return (Map)StepUtil.getTag(getSkLearnTags(), "input_tags");
	}

	public String getPMMLName(){
		return getOptionalString(SkLearn2PMMLFields.PMML_NAME);
	}

	public Step setPMMLName(String pmmlName){
		setattr(SkLearn2PMMLFields.PMML_NAME, pmmlName);

		return this;
	}

	public Map<String, ?> getSkLearnTags(){
		return getOptionalDict(SkLearnFields.SKLEARN_TAGS);
	}

	public String getSkLearnVersion(){
		return getOptionalString(SkLearnFields.SKLEARN_VERSION);
	}

	public Classifier getClassifier(String name){
		return getEstimator(name, Classifier.class);
	}

	public Regressor getRegressor(String name){
		return getEstimator(name, Regressor.class);
	}

	public Estimator getEstimator(String name){
		return getEstimator(name, Estimator.class);
	}

	public <E extends Estimator> E getEstimator(String name, Class<? extends E> clazz){
		return getStep(name, new EstimatorCastFunction<>(clazz));
	}

	public Transformer getTransformer(String name){
		return getTransformer(name, Transformer.class);
	}

	public <E extends Transformer> E getTransformer(String name, Class<? extends E> clazz){
		return getStep(name, new TransformerCastFunction<>(clazz));
	}

	public Step getStep(String name){
		return getStep(name, Step.class);
	}

	public <E extends Step> E getStep(String name, Class<? extends E> clazz){
		return getStep(name, new StepCastFunction<>(clazz));
	}

	public <E extends Step> E getStep(String name, java.util.function.Function<Object, E> castFunction){

		try {
			return get(name, castFunction);
		} catch(SkLearnException se){
			Attribute attribute = new Attribute(this, name);

			throw ensureContext(se, attribute);
		}
	}

	public Estimator getOptionalEstimator(String name){
		return getOptionalStep(name, new EstimatorCastFunction<>(Estimator.class));
	}

	public Transformer getOptionalTransformer(String name){
		return getOptionalStep(name, new TransformerCastFunction<>(Transformer.class));
	}

	public <E extends Step> E getOptionalStep(String name, java.util.function.Function<Object, E> castFunction){

		try {
			return getOptional(name, castFunction);
		} catch(SkLearnException se){
			Attribute attribute = new Attribute(this, name);

			throw ensureContext(se, attribute);
		}
	}

	public <E extends Estimator> List<E> getEstimatorArray(String name, Class<? extends E> clazz){
		return getStepArray(name, new EstimatorCastFunction<>(clazz));
	}

	public <E extends Transformer> List<E> getTransformerArray(String name, Class<? extends E> clazz){
		return getStepArray(name, new TransformerCastFunction<>(clazz));
	}

	public <E extends Step> List<E> getStepArray(String name, java.util.function.Function<Object, E> castFunction){

		try {
			return getArray(name, castFunction);
		} catch(SkLearnException se){
			Attribute attribute = new Attribute(this, name);

			throw ensureContext(se, attribute);
		}
	}

	public List<Classifier> getClassifierList(String name){
		return getEstimatorList(name, Classifier.class);
	}

	public List<Regressor> getRegressorList(String name){
		return getEstimatorList(name, Regressor.class);
	}

	public <E extends Estimator> List<E> getEstimatorList(String name, Class<? extends E> clazz){
		return getStepList(name, new EstimatorCastFunction<>(clazz));
	}

	public <E extends Transformer> List<E> getTransformerList(String name, Class<? extends E> clazz){
		return getStepList(name, new TransformerCastFunction<>(clazz));
	}

	public <E extends Step> List<E> getStepList(String name, java.util.function.Function<Object, E> castFunction){

		try {
			return getList(name, castFunction);
		} catch(SkLearnException se){
			Attribute attribute = new Attribute(this, name);

			throw ensureContext(se, attribute);
		}
	}

	protected <E extends ConversionException> E ensureContext(E exception){
		return ensureContext(exception, this);
	}

	protected <E extends ConversionException> E ensureContext(E exception, Object parentContext){
		Object context = exception.getContext();

		if(context == null){
			exception.setContext(parentContext);
		}

		return exception;
	}
}