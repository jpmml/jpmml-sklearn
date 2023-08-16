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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import hex.genmodel.MojoModel;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureList;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.h2o.Converter;
import org.jpmml.h2o.ConverterFactory;
import org.jpmml.h2o.H2OEncoder;
import org.jpmml.h2o.MojoModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.HasClasses;

public class H2OEstimator extends Estimator implements HasClasses, Encodable {

	private MojoModel mojoModel = null;


	public H2OEstimator(String module, String name){
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
	public int getNumberOfOutputs(){
		String estimatorType = getEstimatorType();

		switch(estimatorType){
			case "classifier":
			case "regressor":
				return 1;
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

	@Override
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
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		String estimatorType = getEstimatorType();

		ClassDictUtil.checkSize(1, names);

		String name = names.get(0);

		switch(estimatorType){
			case "classifier":
				{
					List<?> categories = getClasses();

					DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

					if(name != null){
						DataField dataField = encoder.createDataField(name, OpType.CATEGORICAL, dataType, categories);

						return new CategoricalLabel(dataField);
					} else

					{
						return new CategoricalLabel(dataType, categories);
					}
				}
			case "regressor":
				{
					if(name != null){
						DataField dataField = encoder.createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);

						return new ContinuousLabel(dataField);
					} else

					{
						return new ContinuousLabel(DataType.DOUBLE);
					}
				}
			default:
				throw new IllegalArgumentException(estimatorType);
		}
	}

	@Override
	public Model encodeModel(Schema schema){
		Converter<?> converter = createConverter();

		PMMLEncoder encoder = schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		H2OEncoder h2oEncoder = new H2OEncoder();

		Schema h2oSchema = converter.encodeSchema(h2oEncoder);

		List<? extends Feature> h2oFeatures = h2oSchema.getFeatures();

		List<Feature> reorderedFeatures = new ArrayList<>();

		for(Feature h2oFeature : h2oFeatures){
			String name = h2oFeature.getName();

			Feature feature;

			if(features instanceof FeatureList){
				FeatureList namedFeatures = (FeatureList)features;

				feature = namedFeatures.resolveFeature(name);
			} else

			{
				feature = FeatureUtil.findFeature(features, name);

				if(feature == null){
					int index = Integer.parseInt(name.substring(1)) - 1;

					feature = features.get(index);
				}
			}

			reorderedFeatures.add(feature);
		}

		Schema mojoModelSchema = converter.toMojoModelSchema(new Schema(encoder, label, reorderedFeatures));

		return converter.encodeModel(mojoModelSchema);
	}

	@Override
	public PMML encodePMML(){
		Converter<?> converter = createConverter();

		return converter.encodePMML();
	}

	public String getEstimatorType(){
		return getString("_estimator_type");
	}

	public byte[] getMojoBytes(){
		return get("_mojo_bytes", byte[].class);
	}

	public String getMojoPath(){
		return getString("_mojo_path");
	}

	public H2OEstimator setMojoPath(String mojoPath){
		put("_mojo_path", mojoPath);

		return this;
	}

	private Converter<?> createConverter(){
		MojoModel mojoModel = getMojoModel();

		try {
			ConverterFactory converterFactory = ConverterFactory.newConverterFactory();

			return converterFactory.newConverter(mojoModel);
		} catch(Exception e){
			throw new IllegalArgumentException(e);
		}
	}

	private MojoModel getMojoModel(){

		if(this.mojoModel == null){
			this.mojoModel = loadMojoModel();
		}

		return this.mojoModel;
	}

	private MojoModel loadMojoModel(){
		MojoModel mojoModel;

		try {

			if(containsKey("_mojo_bytes")){
				byte[] mojoBytes = getMojoBytes();

				try(InputStream is = new ByteArrayInputStream(mojoBytes)){
					mojoModel = MojoModelUtil.readFrom(is);
				}
			} else

			{
				String mojoPath = getMojoPath();

				mojoModel = MojoModelUtil.readFrom(new File(mojoPath), false);
			}
		} catch(Exception e){
			throw new IllegalArgumentException(e);
		}

		return mojoModel;
	}
}