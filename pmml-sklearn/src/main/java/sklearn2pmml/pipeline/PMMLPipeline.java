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
import java.util.stream.Collectors;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import numpy.core.NDArrayUtil;
import numpy.core.ScalarUtil;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Header;
import org.dmg.pmml.MiningBuildTask;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.Value;
import org.dmg.pmml.VerificationField;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.visitors.AbstractExtender;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonObject;
import org.jpmml.sklearn.FieldNames;
import org.jpmml.sklearn.SkLearnEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasClassifierOptions;
import sklearn.HasEstimatorEnsemble;
import sklearn.HasNumberOfFeatures;
import sklearn.Initializer;
import sklearn.ScalarLabelUtil;
import sklearn.Step;
import sklearn.Transformer;
import sklearn.pipeline.FeatureUnion;
import sklearn.pipeline.Pipeline;
import sklearn.pipeline.PipelineClassifier;
import sklearn.pipeline.PipelineRegressor;
import sklearn.pipeline.PipelineTransformer;
import sklearn2pmml.decoration.Domain;

public class PMMLPipeline extends Pipeline {

	public PMMLPipeline(){
		this("sklearn2pmml", "PMMLPipeline");
	}

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		logger.warn(ClassDictUtil.formatClass(this) + " should be replaced with " + ClassDictUtil.formatClass(new Pipeline()) + " in nested workflows");

		return super.encodeFeatures(features, encoder);
	}

	public PMML encodePMML(SkLearnEncoder encoder){
		List<? extends Transformer> transformers = getTransformers();
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

		Label label = null;

		if(estimator != null && estimator.isSupervised()){

			if(targetFields == null){
				targetFields = initTargetFields(estimator);
			}

			label = initLabel(estimator, targetFields, encoder);
		}

		List<Feature> features = new ArrayList<>();

		PythonObject featureInitializer = estimator;

		try {
			Transformer transformer = getHead(transformers, estimator);

			if(transformer != null){
				featureInitializer = transformer;

				if(!(transformer instanceof Initializer)){

					if(activeFields == null){
						activeFields = initActiveFields(transformer);
					}

					features = initFeatures(transformer, activeFields, transformer.getOpType(), transformer.getDataType(), encoder);
				}

				features = super.encodeFeatures(features, encoder);
			} else

			if(estimator != null){

				if(activeFields == null){
					activeFields = initActiveFields(estimator);
				}

				features = initFeatures(estimator, activeFields, estimator.getOpType(), estimator.getDataType(), encoder);
			}
		} catch(UnsupportedOperationException uoe){
			throw new IllegalArgumentException("The transformer object of the first step (" + ClassDictUtil.formatClass(featureInitializer) + ") does not specify feature type information", uoe);
		}

		if(estimator == null){
			return encodePMML(header, null, repr, encoder);
		}

		Schema schema = new Schema(encoder, label, features);

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

			Output output = ModelUtil.ensureOutput(finalModel);

			if(predictTransformer != null){
				List<ScalarLabel> scalarLabels;

				if(label instanceof ScalarLabel){
					ScalarLabel scalarLabel = (ScalarLabel)label;

					scalarLabels = Collections.singletonList(scalarLabel);
				} else

				if(label instanceof MultiLabel){
					MultiLabel multiLabel = (MultiLabel)label;

					List<? extends Label> labels = multiLabel.getLabels();

					scalarLabels = labels.stream()
						.map(ScalarLabel.class::cast)
						.collect(Collectors.toList());
				} else

				{
					throw new IllegalArgumentException();
				}

				List<OutputField> predictFields = new ArrayList<>();

				for(ScalarLabel scalarLabel : scalarLabels){
					OutputField predictField = ModelUtil.createPredictedField(FieldNameUtil.create(Estimator.FIELD_PREDICT, scalarLabel.getName()), ScalarLabelUtil.getOpType(scalarLabel), scalarLabel.getDataType())
						.setFinalResult(false);

					output.addOutputFields(predictField);

					predictFields.add(predictField);
				}

				encodeOutput(output, predictFields, predictTransformer, encoder);
			} // End if

			if(predictProbaTransformer != null){
				CategoricalLabel categoricalLabel = (CategoricalLabel)label;

				List<OutputField> predictProbaFields = ModelUtil.createProbabilityFields(DataType.DOUBLE, categoricalLabel.getValues());

				encodeOutput(output, predictProbaFields, predictProbaTransformer, encoder);
			} // End if

			if(applyTransformer != null){
				OutputField nodeIdField = ModelUtil.createEntityIdField(FieldNames.NODE_ID, DataType.INTEGER);

				encodeOutput(output, Collections.singletonList(nodeIdField), applyTransformer, encoder);
			}

			encoder.setModel(model);
		} // End if

		verification:
		if(estimator.isSupervised()){

			if(verification == null){
				logger.warn("Model verification data is not set. Use method \'" + ClassDictUtil.formatMember(this, "verify(X)") + "\' to correct this deficiency");

				break verification;
			}

			int[] activeValuesShape = verification.getActiveValuesShape();
			int[] targetValuesShape = verification.getTargetValuesShape();

			ClassDictUtil.checkShapes(0, activeValuesShape, targetValuesShape);
			ClassDictUtil.checkShapes(1, activeFields.size(), activeValuesShape);

			List<?> activeValues = verification.getActiveValues();
			List<?> targetValues = verification.getTargetValues();

			int[] probabilityValuesShape = null;

			List<? extends Number> probabilityValues = null;

			boolean hasProbabilityValues = verification.hasProbabilityValues();

			if(estimator instanceof Classifier){
				Classifier classifier = (Classifier)estimator;

				hasProbabilityValues &= classifier.hasProbabilityDistribution();
			} else

			{
				hasProbabilityValues = false;
			} // End if

			if(hasProbabilityValues){
				probabilityFields = initProbabilityFields((CategoricalLabel)label);

				probabilityValuesShape = verification.getProbabilityValuesShape();

				ClassDictUtil.checkShapes(0, activeValuesShape, probabilityValuesShape);
				ClassDictUtil.checkShapes(1, probabilityFields.size(), probabilityValuesShape);

				probabilityValues = verification.getProbabilityValues();
			}

			Number precision = verification.getPrecision();
			Number zeroThreshold = verification.getZeroThreshold();

			int rows = activeValuesShape[0];

			Map<VerificationField, List<?>> data = new LinkedHashMap<>();

			if(activeFields != null){

				for(int i = 0; i < activeFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(activeFields.get(i));

					Domain domain = encoder.getDomain(verificationField.requireField());

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(domain, activeValues), rows, activeFields.size(), i));
				}
			} // End if

			if(probabilityFields != null){

				for(int i = 0; i < probabilityFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(probabilityFields.get(i))
						.setPrecision(precision)
						.setZeroThreshold(zeroThreshold);

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(null, probabilityValues), rows, probabilityFields.size(), i));
				}
			} else

			{
				ScalarLabel scalarLabel = (ScalarLabel)label;

				for(int i = 0; i < targetFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(targetFields.get(i));

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

	public PMMLPipeline setSteps(List<Object[]> steps){
		put("steps", steps);

		return this;
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

	private List<String> initActiveFields(Step step){
		int numberOfFeatures = step.getNumberOfFeatures();

		if(numberOfFeatures == HasNumberOfFeatures.UNKNOWN){
			throw new IllegalArgumentException("The transformer object of the first step (" + ClassDictUtil.formatClass(step) + ") does not specify the number of input features");
		}

		List<String> activeFields = makeVariables("x", numberOfFeatures, true);

		logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "active_fields") + "\' is not set. Assuming {} as the names of active fields", activeFields);

		return activeFields;
	}

	private List<String> initProbabilityFields(CategoricalLabel categoricalLabel){
		List<String> probabilityFields = new ArrayList<>();

		List<?> values = categoricalLabel.getValues();
		for(Object value : values){
			probabilityFields.add("probability(" + value + ")"); // XXX
		}

		return probabilityFields;
	}

	private List<String> initTargetFields(Estimator estimator){
		int numberOfOutputs = estimator.getNumberOfOutputs();

		if(numberOfOutputs == -1){
			logger.warn("The estimator object of the final step (" + ClassDictUtil.formatClass(estimator) + ") does not specify the number of outputs. Assuming a single output");

			numberOfOutputs = 1;
		}

		List<String> targetFields = makeVariables("y", numberOfOutputs, false);

		logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "target_fields") + "\' is not set. Assuming {} as the name of target fields", targetFields);

		return targetFields;
	}

	static
	private Label initLabel(Estimator estimator, List<String> targetFields, SkLearnEncoder encoder){
		List<Label> labels = new ArrayList<>();

		MiningFunction miningFunction = estimator.getMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
				{
					List<?> categories = EstimatorUtil.getClasses(estimator);
					Map<String, Map<String, ?>> classExtensions = (Map)estimator.getOption(HasClassifierOptions.OPTION_CLASS_EXTENSIONS, null);

					// XXX
					if(classExtensions != null){
						ClassDictUtil.checkSize(1, targetFields);
					}

					for(int i = 0; i < targetFields.size(); i++){
						String targetField = targetFields.get(i);

						labels.add(initCategoricalLabel(targetField, (targetFields.size() > 1 ? (List<?>)categories.get(i) : categories), classExtensions, encoder));
					}
				}
				break;
			case REGRESSION:
				{
					for(int i = 0; i < targetFields.size(); i++){
						String targetField = targetFields.get(i);

						labels.add(initContinuousLabel(targetField, encoder));
					}
				}
				break;
			case MIXED:
				{
					HasEstimatorEnsemble<?> hasEstimatorEnsemble = (HasEstimatorEnsemble<?>)estimator;

					List<? extends Estimator> estimators = hasEstimatorEnsemble.getEstimators();

					ClassDictUtil.checkSize(targetFields, estimators);

					for(int i = 0; i < targetFields.size(); i++){
						String targetField = targetFields.get(i);

						labels.add((ScalarLabel)initLabel(estimators.get(i), Collections.singletonList(targetField), encoder));
					}
				}
				break;
			default:
				throw new IllegalArgumentException();
		}

		if(labels.size() == 1){
			return labels.get(0);
		} else

		if(labels.size() >= 2){
			return new MultiLabel(labels);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	private CategoricalLabel initCategoricalLabel(String targetField, List<?> categories, Map<String, Map<String, ?>> classExtensions, SkLearnEncoder encoder){
		DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

		DataField dataField = encoder.createDataField(targetField, OpType.CATEGORICAL, dataType, categories);

		List<Visitor> visitors = new ArrayList<>();

		if(classExtensions != null){
			Collection<? extends Map.Entry<String, Map<String, ?>>> entries = classExtensions.entrySet();

			for(Map.Entry<String, Map<String, ?>> entry : entries){
				String name = entry.getKey();

				Map<String, ?> values = entry.getValue();

				Visitor valueExtender = new AbstractExtender(name){

					@Override
					public VisitorAction visit(Value pmmlValue){
						Object value = values.get(pmmlValue.requireValue());

						if(value != null){
							value = ScalarUtil.decode(value);

							addExtension(pmmlValue, ValueUtil.asString(value));
						}

						return super.visit(pmmlValue);
					}
				};

				visitors.add(valueExtender);
			}
		}

		for(Visitor visitor : visitors){
			visitor.applyTo(dataField);
		}

		return new CategoricalLabel(dataField);
	}

	static
	private ContinuousLabel initContinuousLabel(String targetField, SkLearnEncoder encoder){
		DataField dataField = encoder.createDataField(targetField, OpType.CONTINUOUS, DataType.DOUBLE);

		return new ContinuousLabel(dataField);
	}

	static
	private List<Feature> initFeatures(Step step, List<String> activeFields, OpType opType, DataType dataType, SkLearnEncoder encoder){
		List<Feature> result = new ArrayList<>();

		for(String activeField : activeFields){
			DataField dataField = encoder.createDataField(activeField, opType, dataType);

			result.add(new WildcardFeature(encoder, dataField));
		}

		return result;
	}

	static
	private Transformer getHead(List<? extends Transformer> transformers, Estimator estimator){

		if(!transformers.isEmpty()){
			Transformer transformer = transformers.get(0);

			if(transformer instanceof FeatureUnion){
				FeatureUnion featureUnion = (FeatureUnion)transformer;

				return getHead(featureUnion.getTransformers(), null);
			} else

			if(transformer instanceof PipelineTransformer){
				PipelineTransformer pipelineTransformer = (PipelineTransformer)transformer;

				Pipeline pipeline = pipelineTransformer.getPipeline();

				return getHead(pipeline.getTransformers(), null);
			} else

			{
				return transformer;
			}
		} // End if

		if(estimator != null){

			if(estimator instanceof PipelineClassifier){
				PipelineClassifier pipelineClassifier = (PipelineClassifier)estimator;

				Pipeline pipeline = pipelineClassifier.getPipeline();

				return getHead(pipeline.getTransformers(), pipeline.getFinalEstimator());
			} else

			if(estimator instanceof PipelineRegressor){
				PipelineRegressor pipelineRegressor = (PipelineRegressor)estimator;

				Pipeline pipeline = pipelineRegressor.getPipeline();

				return getHead(pipeline.getTransformers(), pipeline.getFinalEstimator());
			} else

			{
				return null;
			}
		}

		return null;
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

	static
	private List<String> makeVariables(String name, int count, boolean indexed){

		if(count <= 0){
			throw new IllegalArgumentException();
		} else

		if(count == 1){
			return Collections.singletonList(name + (indexed ? "1" : ""));
		} else

		{
			List<String> result = new ArrayList<>(count);

			for(int i = 0; i < count; i++){
				result.add(name + String.valueOf(i + 1));
			}

			return result;
		}
	}

	private static final Logger logger = LoggerFactory.getLogger(PMMLPipeline.class);
}