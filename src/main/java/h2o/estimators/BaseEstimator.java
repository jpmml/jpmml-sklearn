/*
 * Copyright (c) 2018 Villu Ruusmann
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
package h2o.estimators;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import hex.genmodel.MojoModel;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.h2o.Converter;
import org.jpmml.h2o.ConverterFactory;
import org.jpmml.h2o.H2OEncoder;
import org.jpmml.h2o.MojoModelUtil;
import org.jpmml.sklearn.FeatureList;
import sklearn.Estimator;
import sklearn.HasClasses;

public class BaseEstimator extends Estimator implements HasClasses {

	private MojoModel mojoModel = null;


	public BaseEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		String estimatorType = getEstimatorType();

		switch(estimatorType){
			case "classifier":
				return MiningFunction.CLASSIFICATION;
			case "regressor":
				return MiningFunction.REGRESSION;
			default:
				throw new IllegalArgumentException(estimatorType);
		}
	}

	@Override
	public List<?> getClasses(){
		MojoModel mojoModel = getMojoModel();

		int responseIdx = mojoModel.getResponseIdx();

		String[] responseValues = mojoModel.getDomainValues(responseIdx);
		if(responseValues == null){
			throw new IllegalArgumentException();
		}

		return Arrays.asList(responseValues);
	}

	public boolean hasProbabilityDistribution(){
		String estimatorType = getEstimatorType();

		switch(estimatorType){
			case "classifier":
				return true;
			case "regressor":
				return false;
			default:
				throw new IllegalArgumentException(estimatorType);
		}
	}

	@Override
	public Model encodeModel(Schema schema){
		MojoModel mojoModel = getMojoModel();

		Converter<?> converter;

		try {
			ConverterFactory converterFactory = ConverterFactory.newConverterFactory();

			converter = converterFactory.newConverter(mojoModel);
		} catch(Exception e){
			throw new IllegalArgumentException(e);
		}

		PMMLEncoder encoder = schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		H2OEncoder h2oEncoder = new H2OEncoder();

		Schema h2oSchema = converter.encodeSchema(h2oEncoder);

		List<? extends Feature> h2oFeatures = h2oSchema.getFeatures();

		List<Feature> sortedFeatures = new ArrayList<>();

		for(Feature h2oFeature : h2oFeatures){
			FieldName name = h2oFeature.getName();

			Feature feature;

			if(features instanceof FeatureList){
				FeatureList namedFeatures = (FeatureList)features;

				feature = namedFeatures.getFeature(name.getValue());
			} else

			{
				int index = Integer.parseInt((name.getValue()).substring(1)) - 1;

				feature = features.get(index);
			}

			sortedFeatures.add(feature);
		}

		Schema mojoModelSchema = converter.toMojoModelSchema(new Schema(encoder, label, sortedFeatures));

		return converter.encodeModel(mojoModelSchema);
	}

	public String getEstimatorType(){
		return getString("_estimator_type");
	}

	public String getMojoPath(){
		return getString("_mojo_path");
	}

	public BaseEstimator setMojoPath(String mojoPath){
		put("_mojo_path", mojoPath);

		return this;
	}

	private MojoModel getMojoModel(){

		if(this.mojoModel == null){
			this.mojoModel = loadMojoModel();
		}

		return this.mojoModel;
	}

	private MojoModel loadMojoModel(){
		String mojoPath = getMojoPath();

		MojoModel mojoModel;

		try {
			mojoModel = MojoModelUtil.readFrom(new File(mojoPath), false);
		} catch(Exception e){
			throw new IllegalArgumentException(e);
		}

		return mojoModel;
	}
}