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
package sklearn;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.python.Castable;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;

abstract
public class Composite extends Step implements Castable, HasFeatureNamesIn, HasHead, HasTail {

	/**
	 * The result of casting this composite step.
	 * A composite step can be logically and physically cast to one Step subclass only.
	 *
	 * The first cast operation freezes the result object.
	 * Subsequent cast operations, if any, must agree with the first one on the Step subclass.
	 *
	 * Furthermore, all cast operations must return the same object instance,
	 * in order to enable the same step graph (not simply an equivalent step graph!) to be created and re-created in different encoding stages.
	 */
	private Object castResult = null;


	public Composite(String module, String name){
		super(module, name);
	}

	abstract
	public boolean hasTransformers();

	abstract
	public List<? extends Transformer> getTransformers();

	abstract
	public boolean hasFinalEstimator();

	abstract
	public Estimator getFinalEstimator();

	abstract
	public <E extends Estimator> E getFinalEstimator(Class<? extends E> clazz);

	@Override
	public List<String> getFeatureNamesIn(){
		Step head = getHead();

		if(head != null){
			return head.getFeatureNamesIn();
		}

		return null;
	}

	@Override
	public int getNumberOfFeatures(){
		Step head = getHead();

		if(head != null){
			return head.getNumberOfFeatures();
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public OpType getOpType(){
		Step head = getHead();

		if(head != null){
			return head.getOpType();
		}

		throw new UnsupportedOperationException();
	}

	@Override
	public DataType getDataType(){
		Step head = getHead();

		if(head != null){
			return head.getDataType();
		}

		throw new UnsupportedOperationException();
	}

	@Override
	public Step getTail(){
		Estimator estimator = getFinalEstimator();

		return StepUtil.getTail(estimator);
	}

	/**
	 * @see Transformer
	 */
	public List<Feature> encodeFeatures(Step parent, List<Feature> features, SkLearnEncoder encoder){
		Step prevParent = getParent();

		try {
			setParent(parent);

			return encodeFeatures(features, encoder);
		} finally {
			setParent(prevParent);
		}
	}

	/**
	 * @see Transformer
	 */
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){

		if(hasTransformers()){
			List<? extends Transformer> transformers = getTransformers();

			for(Transformer transformer : transformers){
				features = transformer.encode(this, features, encoder);
			}
		}

		return features;
	}

	/**
	 * @see Estimator
	 */
	public Model encodeModel(Step parent, Schema schema){
		Step prevParent = getParent();

		try {
			setParent(parent);

			return encodeModel(schema);
		} finally {
			setParent(prevParent);
		}
	}

	/**
	 * @see Estimator
	 */
	@SuppressWarnings("unchecked")
	public Model encodeModel(Schema schema){
		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(hasTransformers()){
			features = encodeFeatures((List<Feature>)features, encoder);

			// Refresh label in case some transformer refined the label-backing field
			label = refreshLabel(label, encoder);

			schema = new Schema(encoder, label, features);
		} // End if

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.encode(this, schema);
		}

		throw new UnsupportedOperationException();
	}

	protected Label refreshLabel(Label label, SkLearnEncoder encoder){
		return label;
	}

	@Override
	public Object castTo(Class<?> clazz){

		if(Objects.equals(Transformer.class, clazz)){
			return toTransformer();
		} else

		if(Objects.equals(Estimator.class, clazz)){
			return toEstimator();
		} else

		if(Objects.equals(Classifier.class, clazz)){
			return toClassifier();
		} else

		if(Objects.equals(Regressor.class, clazz)){
			return toRegressor();
		} else

		if(Objects.equals(Clusterer.class, clazz)){
			return toClusterer();
		}

		return this;
	}

	public Transformer toTransformer(){

		if(hasFinalEstimator()){
			throw new IllegalStateException();
		} // End if

		if(this.castResult == null){
			this.castResult = new CompositeTransformer(this);
		}

		return (Transformer)this.castResult;
	}

	public Estimator toEstimator(){
		Estimator estimator = getFinalEstimator();

		MiningFunction miningFunction = estimator.getMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
				return toClassifier();
			case REGRESSION:
				return toRegressor();
			case CLUSTERING:
				return toClusterer();
			default:
				throw new IllegalArgumentException();
		}
	}

	public Classifier toClassifier(){

		if(this.castResult == null){
			this.castResult = new CompositeClassifier(this);
		}

		return (Classifier)this.castResult;
	}

	public Regressor toRegressor(){

		if(this.castResult == null){
			this.castResult = new CompositeRegressor(this);
		}

		return (Regressor)this.castResult;
	}

	public Clusterer toClusterer(){

		if(this.castResult == null){
			this.castResult = new CompositeClusterer(this);
		}

		return (Clusterer)this.castResult;
	}

	protected List<String> initLabel(List<String> targetFields, SkLearnEncoder encoder){
		Estimator estimator = getFinalEstimator();

		if(estimator != null && estimator.isSupervised()){

			if(targetFields == null){
				targetFields = initTargetFields(estimator);
			}

			encoder.initLabel(estimator, targetFields);
		}

		return targetFields;
	}

	protected List<String> initTargetFields(Estimator estimator){
		return EncodableUtil.generateOutputNames(estimator);
	}

	protected List<String> initFeatures(List<String> activeFields, SkLearnEncoder encoder){
		Step head = getHead();

		try {
			if(head instanceof Transformer){

				if(!(head instanceof Initializer)){

					if(activeFields == null){
						activeFields = initActiveFields(head);
					}

					encoder.initFeatures(head, activeFields);
				}

				// XXX
				List<Feature> features = new ArrayList<>();
				features.addAll(encoder.getFeatures());

				features = encodeFeatures(features, encoder);

				encoder.setFeatures(features);
			} else

			if(head instanceof Estimator){

				if(activeFields == null){
					activeFields = initActiveFields(head);
				}

				encoder.initFeatures(head, activeFields);
			} else

			{
				throw new SkLearnException("The head object (" + ClassDictUtil.formatClass(head)  + ") is not a supported Transformer or Estimator")
					.setSolution("Develop and register a custom JPMML-SkLearn converter");
			}
		} catch(UnsupportedOperationException uoe){
			throw new SkLearnException("The feature initializer object (" + ClassDictUtil.formatClass(head) + ") does not specify feature type information")
				.setSolution("Prepend Domain decorators");
		}

		return activeFields;
	}

	protected List<String> initActiveFields(Step step){
		return EncodableUtil.getOrGenerateFeatureNames(step);
	}
}