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
import hex.genmodel.algos.glm.GlmOrdinalMojoModel;
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
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.OrdinalLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.h2o.Converter;
import org.jpmml.h2o.ConverterFactory;
import org.jpmml.h2o.H2OEncoder;
import org.jpmml.h2o.MojoModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.HasClasses;
import sklearn2pmml.SkLearn2PMMLFields;

public class H2OEstimator extends Estimator implements HasClasses, Encodable {

	private MojoModel mojoModel = null;


	public H2OEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		String estimatorType = getEstimatorType();

		switch(estimatorType){
			case H2OEstimator.TYPE_CLASSIFIER:
				return MiningFunction.CLASSIFICATION;
			case H2OEstimator.TYPE_REGRESSOR:
				return MiningFunction.REGRESSION;
			default:
				throw new IllegalArgumentException(estimatorType);
		}
	}

	@Override
	public boolean isSupervised(){
		String estimatorType = getEstimatorType();

		switch(estimatorType){
			case H2OEstimator.TYPE_CLASSIFIER:
			case H2OEstimator.TYPE_REGRESSOR:
				return true;
			default:
				throw new IllegalArgumentException(estimatorType);
		}
	}

	@Override
	public int getNumberOfOutputs(){
		String estimatorType = getEstimatorType();

		switch(estimatorType){
			case H2OEstimator.TYPE_CLASSIFIER:
			case H2OEstimator.TYPE_REGRESSOR:
				return 1;
			default:
				throw new IllegalArgumentException(estimatorType);
		}
	}

	@Override
	public List<?> getClasses(){
		MojoModel mojoModel = getMojoModel();

		if(hasattr(SkLearn2PMMLFields.PMML_CLASSES)){
			List<?> values = getListLike(SkLearn2PMMLFields.PMML_CLASSES);

			return Classifier.canonicalizeValues(values);
		}

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
			case H2OEstimator.TYPE_CLASSIFIER:
				return true;
			case H2OEstimator.TYPE_REGRESSOR:
				return false;
			default:
				throw new IllegalArgumentException(estimatorType);
		}
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		String estimatorType = getEstimatorType();
		MojoModel mojoModel = getMojoModel();

		ClassDictUtil.checkSize(1, names);

		String name = names.get(0);

		switch(estimatorType){
			case H2OEstimator.TYPE_CLASSIFIER:
				{
					List<?> categories = getClasses();

					OpType opType = OpType.CATEGORICAL;
					DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

					// XXX
					if(mojoModel instanceof GlmOrdinalMojoModel){
						opType = OpType.ORDINAL;
					} // End if

					if(name != null){
						DataField dataField = encoder.createDataField(name, opType, dataType, categories);

						return ScalarLabelUtil.createScalarLabel(dataField);
					} else

					{
						switch(opType){
							case CATEGORICAL:
								return new CategoricalLabel(dataType, categories);
							case ORDINAL:
								return new OrdinalLabel(dataType, categories);
							default:
								throw new IllegalArgumentException();
						}
					}
				}
			case H2OEstimator.TYPE_REGRESSOR:
				{
					if(name != null){
						DataField dataField = encoder.createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);

						return ScalarLabelUtil.createScalarLabel(dataField);
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

		ModelEncoder encoder = schema.getEncoder();
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
		return getEnum("_estimator_type", this::getString, Arrays.asList(H2OEstimator.TYPE_CLASSIFIER, H2OEstimator.TYPE_REGRESSOR));
	}

	public byte[] getMojoBytes(){
		return get("_mojo_bytes", byte[].class);
	}

	public String getMojoPath(){
		return getString("_mojo_path");
	}

	public H2OEstimator setMojoPath(String mojoPath){
		setattr("_mojo_path", mojoPath);

		return this;
	}

	private Converter<?> createConverter(){
		MojoModel mojoModel = getMojoModel();

		try {
			ConverterFactory converterFactory = ConverterFactory.newConverterFactory();

			return converterFactory.newConverter(mojoModel);
		} catch(Exception e){
			throw new SkLearnException("Failed to create H2O.ai converter", e);
		}
	}

	private MojoModel getMojoModel(){

		if(this.mojoModel == null){
			this.mojoModel = loadMojoModel();
		}

		return this.mojoModel;
	}

	private MojoModel loadMojoModel(){

		if(hasattr("_mojo_bytes")){
			byte[] mojoBytes = getMojoBytes();

			try(InputStream is = new ByteArrayInputStream(mojoBytes)){
				return MojoModelUtil.readFrom(is);
			} catch(Exception e){
				throw new SkLearnException("Failed to load H2O.ai MOJO object", e);
			}
		} else

		{
			String mojoPath = getMojoPath();

			try {
				return MojoModelUtil.readFrom(new File(mojoPath), false);
			} catch(Exception e){
				throw new SkLearnException("Failed to load H2O.ai MOJO object", e);
			}
		}
	}

	private static final String TYPE_CLASSIFIER = "classifier";
	private static final String TYPE_REGRESSOR = "regressor";
}