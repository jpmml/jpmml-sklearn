/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn2pmml.pipeline;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Header;
import org.dmg.pmml.MiningBuildTask;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.VerificationField;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasNumberOfOutputs;
import sklearn.Initializer;
import sklearn.Step;
import sklearn.StepUtil;
import sklearn.Transformer;
import sklearn.pipeline.SkLearnPipeline;
import sklearn2pmml.decoration.Domain;

public class PMMLPipeline extends SkLearnPipeline implements Encodable {

	public PMMLPipeline(){
		this("sklearn2pmml", "PMMLPipeline");
	}

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		logger.warn(ClassDictUtil.formatClass(this) + " should be replaced with " + ClassDictUtil.formatClass(new SkLearnPipeline()) + " in nested workflows");

		return super.encodeFeatures(features, encoder);
	}

	@Override
	public Model encodeModel(Schema schema){
		return super.encodeModel(schema);
	}

	@Override
	public PMML encodePMML(){
		SkLearnEncoder encoder = new SkLearnEncoder();

		Estimator estimator = null;

		if(hasFinalEstimator()){
			estimator = getFinalEstimator();
		}

		Map<?, ?> header = getHeader();
		Transformer predictTransformer = getPredictTransformer();
		Transformer predictProbaTransformer = getPredictProbaTransformer();
		Transformer applyTransformer = getApplyTransformer();

		List<String> activeFields = getActiveFields();
		List<String> probabilityFields = null;
		List<String> targetFields = getTargetFields();
		String repr = getRepr();
		Verification verification = getVerification();

		if(estimator != null && estimator.isSupervised()){

			if(targetFields == null){
				targetFields = initTargetFields(estimator);
			}

			encoder.initLabel(estimator, targetFields);
		}

		Step featureInitializer = estimator;

		try {
			Transformer transformer = getHead();

			if(transformer != null){
				featureInitializer = transformer;

				if(!(transformer instanceof Initializer)){

					if(activeFields == null){
						activeFields = initActiveFields(transformer);
					}

					encoder.initFeatures(transformer, activeFields);
				}

				// XXX
				List<Feature> features = new ArrayList<>();
				features.addAll(encoder.getFeatures());

				features = super.encodeFeatures(features, encoder);

				encoder.setFeatures(features);
			} else

			if(estimator != null){

				if(activeFields == null){
					activeFields = initActiveFields(estimator);
				}

				encoder.initFeatures(estimator, activeFields);
			}
		} catch(UnsupportedOperationException uoe){
			throw new IllegalArgumentException("The transformer object of the first step (" + ClassDictUtil.formatClass(featureInitializer) + ") does not specify feature type information", uoe);
		}

		if(estimator == null){
			return encodePMML(header, null, repr, encoder);
		}

		Schema schema = encoder.createSchema();

		Model model = estimator.encode(schema);

		encoder.setModel(model);

		if(!estimator.hasFeatureImportances()){
			List<? extends Number> featureImportances = getPMMLFeatureImportances();

			if(featureImportances != null){
				ClassDictUtil.checkSize(activeFields, featureImportances);

				for(int i = 0; i < activeFields.size(); i++){
					String activeField = activeFields.get(i);
					Number featureImportance = featureImportances.get(i);

					DataField dataField = encoder.getDataField(activeField);
					if(dataField == null){
						throw new IllegalArgumentException("Field " + activeField + " is undefined");
					}

					Feature feature = new WildcardFeature(encoder, dataField);

					encoder.addFeatureImportance(model, feature, featureImportance);
				}
			}
		} // End if

		if((predictTransformer != null) || (predictProbaTransformer != null) || (applyTransformer != null)){
			Model finalModel = MiningModelUtil.getFinalModel(model);

			// XXX
			encoder.setModel(finalModel);

			Label label = schema.getLabel();

			Output output = ModelUtil.ensureOutput(finalModel);

			if(predictTransformer != null){
				List<ScalarLabel> scalarLabels = ScalarLabelUtil.toScalarLabels(label);

				List<OutputField> predictFields = new ArrayList<>();

				for(ScalarLabel scalarLabel : scalarLabels){
					OutputField predictField = ModelUtil.createPredictedField(FieldNameUtil.create(Estimator.FIELD_PREDICT, scalarLabel.getName()), scalarLabel.getOpType(), scalarLabel.getDataType())
						.setFinalResult(false);

					output.addOutputFields(predictField);

					predictFields.add(predictField);
				}

				encodeOutput(output, predictFields, predictTransformer, encoder);
			} // End if

			if(predictProbaTransformer != null){
				CategoricalLabel categoricalLabel = (CategoricalLabel)label;

				List<OutputField> predictProbaFields = estimator.createPredictProbaFields(DataType.DOUBLE, categoricalLabel);

				encodeOutput(output, predictProbaFields, predictProbaTransformer, encoder);
			} // End if

			if(applyTransformer != null){
				OutputField applyField = estimator.createApplyField(DataType.INTEGER);

				encodeOutput(output, Collections.singletonList(applyField), applyTransformer, encoder);
			}

			encoder.setModel(model);
		} // End if

		verification:
		if(estimator.isSupervised()){

			if(verification == null){
				logger.warn("Model verification data is not set. Use method \'" + ClassDictUtil.formatMember(this, "verify(X)") + "\' to correct this deficiency");

				break verification;
			}

			Label label = schema.getLabel();

			List<?> activeValues = verification.getActiveValues();
			int[] activeValuesShape = verification.getActiveValuesShape();

			ClassDictUtil.checkShapes(1, activeFields.size(), activeValuesShape);

			int rows = activeValuesShape[0];

			Map<VerificationField, List<?>> data = new LinkedHashMap<>();

			if(activeFields != null){

				for(int i = 0; i < activeFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(activeFields.get(i));

					Domain domain = encoder.getDomain(verificationField.requireField());

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(domain, activeValues), rows, activeFields.size(), i));
				}
			}

			Number precision = verification.getPrecision();
			Number zeroThreshold = verification.getZeroThreshold();

			List<ScalarLabel> scalarLabels = ScalarLabelUtil.toScalarLabels(label);

			boolean hasProbabilityValues = verification.hasProbabilityValues();

			if(estimator instanceof Classifier){
				Classifier classifier = (Classifier)estimator;

				hasProbabilityValues &= classifier.hasProbabilityDistribution();
			} else

			{
				hasProbabilityValues = false;
			} // End if

			if(hasProbabilityValues){
				List<? extends Number> probabilityValues = verification.getProbabilityValues();
				int[] probabilityValuesShape = verification.getProbabilityValuesShape();

				ClassDictUtil.checkShapes(0, activeValuesShape, probabilityValuesShape);

				// XXX
				ClassDictUtil.checkSize(1, scalarLabels);

				ScalarLabel scalarLabel = scalarLabels.get(0);

				probabilityFields = initProbabilityFields((CategoricalLabel)scalarLabel);

				ClassDictUtil.checkShapes(1, probabilityFields.size(), probabilityValuesShape);

				for(int i = 0; i < probabilityFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(probabilityFields.get(i))
						.setPrecision(precision)
						.setZeroThreshold(zeroThreshold);

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(null, probabilityValues), rows, probabilityFields.size(), i));
				}
			} else

			{
				List<?> targetValues = verification.getTargetValues();
				int[] targetValuesShape = verification.getTargetValuesShape();

				ClassDictUtil.checkShapes(0, activeValuesShape, targetValuesShape);

				ClassDictUtil.checkSize(targetFields, scalarLabels);

				for(int i = 0; i < targetFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(targetFields.get(i));

					ScalarLabel scalarLabel = scalarLabels.get(i);

					DataType dataType = scalarLabel.getDataType();
					switch(dataType){
						case DOUBLE:
						case FLOAT:
							verificationField
								.setPrecision(precision)
								.setZeroThreshold(zeroThreshold);
							break;
						default:
							break;
					}

					Domain domain = encoder.getDomain(verificationField.requireField());

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(domain, targetValues), rows, targetFields.size(), i));
				}
			}

			model.setModelVerification(ModelUtil.createModelVerification(data));
		}

		return encodePMML(header, model, repr, encoder);
	}

	private PMML encodePMML(Map<?, ?> header, Model model, String repr, SkLearnEncoder encoder){
		PMML pmml = encoder.encodePMML(model);

		if(header != null){
			Header pmmlHeader = pmml.requireHeader();

			pmmlHeader.setCopyright((String)header.get("copyright"));
			pmmlHeader.setDescription((String)header.get("description"));
			pmmlHeader.setModelVersion((String)header.get("modelVersion"));
		} // End if

		if(repr != null){
			MiningBuildTask miningBuildTask = new MiningBuildTask()
				.addExtensions(PMMLUtil.createExtension("repr", (Object)repr));

			pmml.setMiningBuildTask(miningBuildTask);
		}

		return pmml;
	}

	private void encodeOutput(Output output, List<OutputField> outputFields, Transformer transformer, SkLearnEncoder encoder){
		SkLearnEncoder outputEncoder = new SkLearnEncoder();

		Model model = encoder.getModel();
		if(model != null){
			outputEncoder.setModel(model);
		}

		List<Feature> features = new ArrayList<>();

		for(OutputField outputField : outputFields){
			DataField dataField = outputEncoder.createDataField(outputField.requireName(), outputField.requireOpType(), outputField.requireDataType());

			features.add(new WildcardFeature(outputEncoder, dataField));
		}

		transformer.encode(features, outputEncoder);

		Collection<DerivedField> derivedFields = (outputEncoder.getDerivedFields()).values();

		for(Iterator<DerivedField> it = derivedFields.iterator(); it.hasNext(); ){
			DerivedField derivedField = it.next();

			OutputField outputField;

			if(derivedField instanceof DerivedOutputField){
				DerivedOutputField derivedOutputField = (DerivedOutputField)derivedField;

				outputField = derivedOutputField.getOutputField();
			} else

			{
				outputField = new OutputField(derivedField.requireName(), derivedField.requireOpType(), derivedField.requireDataType())
					.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
					.setFinalResult(!it.hasNext())
					.setExpression(derivedField.requireExpression());
			}

			output.addOutputFields(outputField);
		}

		Map<String, DefineFunction> defineFunctions = outputEncoder.getDefineFunctions();

		for(DefineFunction defineFunction : defineFunctions.values()){
			encoder.addDefineFunction(defineFunction);
		}
	}

	@Override
	public List<Object[]> getSteps(){
		return super.getSteps();
	}

	@Override
	public PMMLPipeline setSteps(List<Object[]> steps){
		return (PMMLPipeline)super.setSteps(steps);
	}

	public Map<?, ?> getHeader(){
		return getOptional("header", Map.class);
	}

	public List<? extends Number> getPMMLFeatureImportances(){

		if(!containsKey("pmml_feature_importances_")){
			return null;
		}

		return getNumberArray("pmml_feature_importances_");
	}

	public Transformer getPredictTransformer(){
		return getTransformer("predict_transformer");
	}

	public Transformer getPredictProbaTransformer(){
		return getTransformer("predict_proba_transformer");
	}

	public Transformer getApplyTransformer(){
		return getTransformer("apply_transformer");
	}

	private Transformer getTransformer(String key){
		return getOptional(key, Transformer.class);
	}

	public List<String> getActiveFields(){

		if(!containsKey("active_fields")){
			return null;
		}

		return getListLike("active_fields", String.class);
	}

	public PMMLPipeline setActiveFields(List<String> activeFields){
		put("active_fields", NDArrayUtil.toArray(activeFields));

		return this;
	}

	public List<String> getTargetFields(){

		// SkLearn2PMML 0.24.3
		if(containsKey("target_field")){
			return Collections.singletonList(getOptionalString("target_field"));
		} // End if

		// SkLearn2PMML 0.25+
		if(!containsKey("target_fields")){
			return null;
		}

		return getListLike("target_fields", String.class);
	}

	public PMMLPipeline setTargetFields(List<String> targetFields){
		put("target_fields", NDArrayUtil.toArray(targetFields));

		return this;
	}

	public String getRepr(){
		return getOptionalString("repr_");
	}

	public PMMLPipeline setRepr(String repr){
		put("repr_", repr);

		return this;
	}

	public Verification getVerification(){
		return getOptional("verification", Verification.class);
	}

	public PMMLPipeline setVerification(Verification verification){
		put("verification", verification);

		return this;
	}

	private List<String> initProbabilityFields(CategoricalLabel categoricalLabel){
		List<String> probabilityFields = new ArrayList<>();

		List<?> values = categoricalLabel.getValues();
		for(Object value : values){
			probabilityFields.add(FieldNameUtil.create(Classifier.FIELD_PROBABILITY, value));
		}

		return probabilityFields;
	}

	private List<String> initTargetFields(Estimator estimator){
		List<String> result = Collections.singletonList("y");

		int numberOfOutputs = estimator.getNumberOfOutputs();
		if(numberOfOutputs != HasNumberOfOutputs.UNKNOWN){
			result = EstimatorUtil.generateOutputNames(estimator);
		}

		logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "target_fields") + "\' is not set. Assuming {} as the name(s) of the target field(s)", result);

		return result;
	}

	private List<String> initActiveFields(Step step){
		List<String> result = StepUtil.getOrGenerateFeatureNames(step);

		logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "active_fields") + "\' is not set. Assuming {} as the names of active fields", result);

		return result;
	}

	static
	private List<?> cleanValues(Domain domain, List<?> values){
		Function<Object, Object> function = new Function<Object, Object>(){

			@Override
			public Object apply(Object value){
				Domain.checkValue(value);

				if(ValueUtil.isNaN(value)){
					return null;
				}

				return value;
			}
		};

		return Lists.transform(values, function);
	}

	private static final Logger logger = LoggerFactory.getLogger(PMMLPipeline.class);
}