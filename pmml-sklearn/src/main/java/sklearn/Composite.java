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

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.python.Castable;
import org.jpmml.sklearn.SkLearnEncoder;

abstract
public class Composite extends Step implements Castable, HasFeatureNamesIn, HasHead {

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

		if(hasTransformers()){
			List<? extends Transformer> transformers = getTransformers();

			for(Transformer transformer : transformers){
				return transformer.getSkLearnFeatureNamesIn();
			}
		} // End if

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.getSkLearnFeatureNamesIn();
		}

		return null;
	}

	@Override
	public int getNumberOfFeatures(){

		if(hasTransformers()){
			List<? extends Transformer> transformers = getTransformers();

			return StepUtil.getNumberOfFeatures(transformers);
		} // End if

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.getNumberOfFeatures();
		}

		return HasNumberOfFeatures.UNKNOWN;
	}

	@Override
	public OpType getOpType(){

		if(hasTransformers()){
			List<? extends Transformer> transformers = getTransformers();

			for(Transformer transformer : transformers){
				return transformer.getOpType();
			}
		} // End if

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.getOpType();
		}

		throw new UnsupportedOperationException();
	}

	@Override
	public DataType getDataType(){

		if(hasTransformers()){
			List<? extends Transformer> transformers = getTransformers();

			for(Transformer transformer : transformers){
				return transformer.getDataType();
			}
		} // End if

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.getDataType();
		}

		throw new UnsupportedOperationException();
	}

	/**
	 * @see Transformer
	 */
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){

		if(!hasTransformers()){
			return features;
		}

		List<? extends Transformer> transformers = getTransformers();
		for(Transformer transformer : transformers){
			features = transformer.encode(features, encoder);
		}

		return features;
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

			return estimator.encode(schema);
		}

		throw new UnsupportedOperationException();
	}

	protected Label refreshLabel(Label label, SkLearnEncoder encoder){
		return label;
	}

	@Override
	public Object castTo(Class<?> clazz){

		if((Transformer.class).equals(clazz)){
			return toTransformer();
		} else

		if((Estimator.class).equals(clazz)){
			return toEstimator();
		} else

		if((Classifier.class).equals(clazz)){
			return toClassifier();
		} else

		if((Regressor.class).equals(clazz)){
			return toRegressor();
		}

		return this;
	}

	public Transformer toTransformer(){

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			if(estimator != null){
				throw new IllegalArgumentException("The pipeline ends with an estimator object");
			}
		}

		return new CompositeTransformer(this);
	}

	public Estimator toEstimator(){
		Estimator estimator = getFinalEstimator();

		if(estimator instanceof Classifier){
			return toClassifier();
		} else

		if(estimator instanceof Regressor){
			return toRegressor();
		} else

		if(estimator instanceof Clusterer){
			return toClusterer();
		}

		throw new IllegalArgumentException();
	}

	public Classifier toClassifier(){
		return new CompositeClassifier(this);
	}

	public Regressor toRegressor(){
		return new CompositeRegressor(this);
	}

	public Clusterer toClusterer(){
		return new CompositeClusterer(this);
	}
}