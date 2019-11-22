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
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.PyClassDict;
import org.jpmml.sklearn.SkLearnEncoder;

abstract
public class Composite extends PyClassDict implements HasNumberOfFeatures, HasType {

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

	@Override
	public int getNumberOfFeatures(){

		if(hasTransformers()){
			List<? extends Transformer> transformers = getTransformers();

			for(Transformer transformer : transformers){
				return TransformerUtil.getNumberOfFeatures(transformer);
			}
		} // End if

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.getNumberOfFeatures();
		}

		return -1;
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
			int numberOfFeatures = TransformerUtil.getNumberOfFeatures(transformer);

			if(numberOfFeatures > -1){
				ClassDictUtil.checkSize(numberOfFeatures, features);
			}

			features = transformer.updateAndEncodeFeatures(features, encoder);
		}

		return features;
	}

	/**
	 * @see Estimator
	 */
	public Model encodeModel(Schema schema){
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(!hasFinalEstimator()){
			throw new UnsupportedOperationException();
		}

		Estimator estimator = getFinalEstimator();

		// XXX
		if(hasTransformers()){
			Feature feature = features.get(0);

			SkLearnEncoder encoder = (SkLearnEncoder)feature.getEncoder();

			features = encodeFeatures((List)features, encoder);

			schema = new Schema(label, features);
		}

		return estimator.encodeModel(schema);
	}
}