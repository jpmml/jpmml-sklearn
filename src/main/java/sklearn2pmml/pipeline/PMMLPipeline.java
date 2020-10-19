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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import h2o.estimators.BaseEstimator;
import numpy.core.ScalarUtil;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Extension;
import org.dmg.pmml.FieldName;
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
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.visitors.AbstractExtender;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PythonObject;
import org.jpmml.sklearn.SkLearnEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Classifier;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasClassifierOptions;
import sklearn.HasNumberOfFeatures;
import sklearn.Initializer;
import sklearn.Step;
import sklearn.StepUtil;
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
				targetFields = initTargetFields();
			}

			ClassDictUtil.checkSize(1, targetFields);

			String targetField = targetFields.get(0);

			MiningFunction miningFunction = estimator.getMiningFunction();
			switch(miningFunction){
				case CLASSIFICATION:
					{
						List<?> categories = ClassifierUtil.getClasses(estimator);
						Map<String, Map<String, ?>> classExtensions = (Map)estimator.getOption(HasClassifierOptions.OPTION_CLASS_EXTENSIONS, null);

						DataType dataType = TypeUtil.getDataType(categories, DataType.STRING);

						DataField dataField = encoder.createDataField(FieldName.create(targetField), OpType.CATEGORICAL, dataType, categories);

						List<Visitor> visitors = new ArrayList<>();

						if(classExtensions != null){
							Collection<? extends Map.Entry<String, Map<String, ?>>> entries = classExtensions.entrySet();

							for(Map.Entry<String, Map<String, ?>> entry : entries){
								String name = entry.getKey();

								Map<String, ?> values = entry.getValue();

								Visitor valueExtender = new AbstractExtender(name){

									@Override
									public VisitorAction visit(Value pmmlValue){
										Object value = values.get(pmmlValue.getValue());

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

						label = new CategoricalLabel(dataField);
					}
					break;
				case REGRESSION:
					{
						DataField dataField = encoder.createDataField(FieldName.create(targetField), OpType.CONTINUOUS, DataType.DOUBLE);

						label = new ContinuousLabel(dataField);
					}
					break;
				default:
					throw new IllegalArgumentException();
			}
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

					features = initFeatures(activeFields, transformer.getOpType(), transformer.getDataType(), encoder);
				}

				features = super.encodeFeatures(features, encoder);
			} else

			if(estimator != null){

				if(activeFields == null){
					activeFields = initActiveFields(estimator);
				}

				features = initFeatures(activeFields, estimator.getOpType(), estimator.getDataType(), encoder);
			}
		} catch(UnsupportedOperationException uoe){
			throw new IllegalArgumentException("The transformer object of the first step (" + ClassDictUtil.formatClass(featureInitializer) + ") does not specify feature type information", uoe);
		}

		if(estimator == null){
			return encodePMML(null, repr, encoder);
		}

		StepUtil.checkNumberOfFeatures(estimator, features);

		Schema schema = new Schema(encoder, label, features);

		Model model = estimator.encode(schema);

		if((predictTransformer != null) || (predictProbaTransformer != null) || (applyTransformer != null)){
			Model finalModel = MiningModelUtil.getFinalModel(model);

			Output output = ModelUtil.ensureOutput(finalModel);

			if(predictTransformer != null){
				FieldName name = FieldNameUtil.create("predict", label.getName());

				OutputField predictField;

				if(label instanceof ContinuousLabel){
					predictField = ModelUtil.createPredictedField(name, OpType.CONTINUOUS, label.getDataType())
						.setFinalResult(false);
				} else

				if(label instanceof CategoricalLabel){
					predictField = ModelUtil.createPredictedField(name, OpType.CATEGORICAL, label.getDataType())
						.setFinalResult(false);
				} else

				{
					throw new IllegalArgumentException();
				}

				output.addOutputFields(predictField);

				encodeOutput(output, Collections.singletonList(predictField), predictTransformer, encoder);
			} // End if

			if(predictProbaTransformer != null){
				CategoricalLabel categoricalLabel = (CategoricalLabel)label;

				List<OutputField> predictProbaFields = ModelUtil.createProbabilityFields(DataType.DOUBLE, categoricalLabel.getValues());

				encodeOutput(output, predictProbaFields, predictProbaTransformer, encoder);
			} // End if

			if(applyTransformer != null){
				OutputField nodeIdField = ModelUtil.createEntityIdField(FieldName.create("nodeId"))
					.setDataType(DataType.INTEGER);

				encodeOutput(output, Collections.singletonList(nodeIdField), applyTransformer, encoder);
			}
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

			if(estimator instanceof BaseEstimator){
				BaseEstimator baseEstimator = (BaseEstimator)estimator;

				hasProbabilityValues &= baseEstimator.hasProbabilityDistribution();
			} else

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
					VerificationField verificationField = ModelUtil.createVerificationField(FieldName.create(activeFields.get(i)));

					Domain domain = encoder.getDomain(verificationField.getField());

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(domain, activeValues), rows, activeFields.size(), i));
				}
			} // End if

			if(probabilityFields != null){

				for(int i = 0; i < probabilityFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(FieldName.create(probabilityFields.get(i)))
						.setPrecision(precision)
						.setZeroThreshold(zeroThreshold);

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(null, probabilityValues), rows, probabilityFields.size(), i));
				}
			} else

			{
				for(int i = 0; i < targetFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(FieldName.create(targetFields.get(i)));

					DataType dataType = label.getDataType();
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

					Domain domain = encoder.getDomain(verificationField.getField());

					data.put(verificationField, CMatrixUtil.getColumn(cleanValues(domain, targetValues), rows, targetFields.size(), i));
				}
			}

			model.setModelVerification(ModelUtil.createModelVerification(data));
		}

		return encodePMML(model, repr, encoder);
	}

	private PMML encodePMML(Model model, String repr, SkLearnEncoder encoder){
		PMML pmml = encoder.encodePMML(model);

		if(repr != null){
			Extension extension = new Extension()
				.addContent(repr);

			MiningBuildTask miningBuildTask = new MiningBuildTask()
				.addExtensions(extension);

			pmml.setMiningBuildTask(miningBuildTask);
		}

		return pmml;
	}

	private void encodeOutput(Output output, List<OutputField> outputFields, Transformer transformer, SkLearnEncoder encoder){
		SkLearnEncoder outputEncoder = new SkLearnEncoder();

		List<Feature> features = new ArrayList<>();

		for(OutputField outputField : outputFields){
			DataField dataField = outputEncoder.createDataField(outputField.getName(), outputField.getOpType(), outputField.getDataType());

			features.add(new WildcardFeature(outputEncoder, dataField));
		}

		transformer.encode(features, outputEncoder);

		Map<FieldName, DerivedField> derivedFields = outputEncoder.getDerivedFields();

		for(DerivedField derivedField : derivedFields.values()){
			OutputField outputField = new OutputField(derivedField.getName(), derivedField.getOpType(), derivedField.getDataType())
				.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
				.setExpression(derivedField.getExpression());

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
		put("active_fields", toArray(activeFields));

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
		put("target_fields", toArray(targetFields));

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

		List<String> activeFields = new ArrayList<>(numberOfFeatures);

		for(int i = 0, max = numberOfFeatures; i < max; i++){
			activeFields.add("x" + String.valueOf(i + 1));
		}

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

	private List<String> initTargetFields(){
		String targetField = "y";

		logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "target_fields") + "\' is not set. Assuming {} as the name of the target field", targetField);

		return Collections.singletonList(targetField);
	}

	static
	private List<Feature> initFeatures(List<String> activeFields, OpType opType, DataType dataType, SkLearnEncoder encoder){
		List<Feature> result = new ArrayList<>();

		for(String activeField : activeFields){
			DataField dataField = encoder.createDataField(FieldName.create(activeField), opType, dataType);

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

	private static final Logger logger = LoggerFactory.getLogger(PMMLPipeline.class);
}