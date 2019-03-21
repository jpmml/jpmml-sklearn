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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import h2o.estimators.BaseEstimator;
import numpy.core.NDArray;
import numpy.core.ScalarUtil;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
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
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.True;
import org.dmg.pmml.Value;
import org.dmg.pmml.VerificationField;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.converter.visitors.AbstractExtender;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.PyClassDict;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TupleUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Classifier;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasClassifierOptions;
import sklearn.HasEstimator;
import sklearn.HasNumberOfFeatures;
import sklearn.Initializer;
import sklearn.Transformer;
import sklearn.TransformerUtil;
import sklearn.TypeUtil;
import sklearn.pipeline.Pipeline;

public class PMMLPipeline extends Pipeline implements HasEstimator<Estimator> {

	public PMMLPipeline(){
		this("sklearn2pmml", "PMMLPipeline");
	}

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	public PMML encodePMML(){
		List<? extends Transformer> transformers = getTransformers();
		Estimator estimator = getEstimator();
		Transformer predictTransformer = getPredictTransformer();
		Transformer predictProbaTransformer = getPredictProbaTransformer();
		Transformer applyTransformer = getApplyTransformer();
		List<String> activeFields = getActiveFields();
		List<String> probabilityFields = null;
		List<String> targetFields = getTargetFields();
		String repr = getRepr();
		Verification verification = getVerification();

		SkLearnEncoder encoder = new SkLearnEncoder();

		Label label = null;

		if(estimator.isSupervised()){
			String targetField = null;

			if(targetFields != null){
				ClassDictUtil.checkSize(1, targetFields);

				targetField = targetFields.get(0);
			} // End if

			if(targetField == null){
				targetField = "y";

				logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "target_fields") + "\' is not set. Assuming {} as the name of the target field", targetField);
			}

			MiningFunction miningFunction = estimator.getMiningFunction();
			switch(miningFunction){
				case CLASSIFICATION:
					{
						List<?> classes = ClassifierUtil.getClasses(estimator);
						Map<String, Map<String, ?>> classExtensions = (Map)estimator.getOption(HasClassifierOptions.OPTION_CLASS_EXTENSIONS, null);

						DataType dataType = TypeUtil.getDataType(classes, DataType.STRING);

						List<String> categories = ClassifierUtil.formatTargetCategories(classes);

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

											addExtension(pmmlValue, ValueUtil.formatValue(value));
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

		PyClassDict featureInitializer = estimator;

		try {
			Transformer transformer = TransformerUtil.getHead(transformers);

			if(transformer != null){
				featureInitializer = transformer;

				if(!(transformer instanceof Initializer)){
					features = initFeatures(transformer, transformer.getOpType(), transformer.getDataType(), encoder);
				}

				features = encodeFeatures(features, encoder);
			} else

			{
				features = initFeatures(estimator, estimator.getOpType(), estimator.getDataType(), encoder);
			}
		} catch(UnsupportedOperationException uoe){
			throw new IllegalArgumentException("The first transformer or estimator object (" + ClassDictUtil.formatClass(featureInitializer) + ") does not specify feature type information", uoe);
		}

		int numberOfFeatures = estimator.getNumberOfFeatures();
		if(numberOfFeatures > -1){
			ClassDictUtil.checkSize(numberOfFeatures, features);
		}

		Schema schema = new Schema(label, features);

		Model model = estimator.encodeModel(schema);

		if(predictTransformer != null){
			Output output;

			if(model instanceof MiningModel){
				MiningModel miningModel = (MiningModel)model;

				Model finalModel = getFinalModel(miningModel);

				output = ModelUtil.ensureOutput(finalModel);
			} else

			{
				output = ModelUtil.ensureOutput(model);
			}

			FieldName name = FieldName.create("predict(" + (label.getName()).getValue() + ")");

			OutputField predictField;

			if(label instanceof ContinuousLabel){
				predictField = ModelUtil.createPredictedField(name, label.getDataType(), OpType.CONTINUOUS)
					.setFinalResult(false);
			} else

			if(label instanceof CategoricalLabel){
				predictField = ModelUtil.createPredictedField(name, label.getDataType(), OpType.CATEGORICAL)
					.setFinalResult(false);
			} else

			{
				throw new IllegalArgumentException();
			}

			output.addOutputFields(predictField);

			encodeOutput(predictTransformer, model, Collections.singletonList(predictField));
		} // End if

		if(predictProbaTransformer != null){
			CategoricalLabel categoricalLabel = (CategoricalLabel)label;

			List<OutputField> predictProbaFields = ModelUtil.createProbabilityFields(DataType.DOUBLE, categoricalLabel.getValues());

			encodeOutput(predictProbaTransformer, model, predictProbaFields);
		} // End if

		if(applyTransformer != null){
			OutputField nodeIdField = ModelUtil.createEntityIdField(FieldName.create("nodeId"))
				.setDataType(DataType.INTEGER);

			encodeOutput(applyTransformer, model, Collections.singletonList(nodeIdField));
		} // End if

		verification:
		if(estimator.isSupervised()){

			if(verification == null){
				logger.warn("Model verification data is not set. Use method '" + ClassDictUtil.formatMember(this, "verify(X)") + "' to correct this deficiency");

				break verification;
			} // End if

			if(activeFields == null){
				throw new IllegalArgumentException();
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
				probabilityValuesShape = verification.getProbabilityValuesShape();

				probabilityFields = new ArrayList<>();

				CategoricalLabel categoricalLabel = (CategoricalLabel)label;

				List<String> values = categoricalLabel.getValues();
				for(String value : values){
					probabilityFields.add("probability(" + value + ")"); // XXX
				}

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

					data.put(verificationField, CMatrixUtil.getColumn(activeValues, rows, activeFields.size(), i));
				}
			} // End if

			if(probabilityFields != null){

				for(int i = 0; i < probabilityFields.size(); i++){
					VerificationField verificationField = ModelUtil.createVerificationField(FieldName.create(probabilityFields.get(i)))
						.setPrecision(precision.doubleValue())
						.setZeroThreshold(zeroThreshold.doubleValue());

					data.put(verificationField, CMatrixUtil.getColumn(probabilityValues, rows, probabilityFields.size(), i));
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
								.setPrecision(precision.doubleValue())
								.setZeroThreshold(zeroThreshold.doubleValue());
							break;
						default:
							break;
					}

					data.put(verificationField, CMatrixUtil.getColumn(targetValues, rows, targetFields.size(), i));
				}
			}

			model.setModelVerification(ModelUtil.createModelVerification(data));
		}

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

	private List<Feature> initFeatures(PyClassDict object, OpType opType, DataType dataType, SkLearnEncoder encoder){
		List<String> activeFields = getActiveFields();

		if(activeFields == null){
			int numberOfFeatures = -1;

			if(object instanceof HasNumberOfFeatures){
				HasNumberOfFeatures hasNumberOfFeatures = (HasNumberOfFeatures)object;

				numberOfFeatures = hasNumberOfFeatures.getNumberOfFeatures();
			} // End if

			if(numberOfFeatures < 0){
				throw new IllegalArgumentException("The first transformer or estimator object (" + ClassDictUtil.formatClass(object) + ") does not specify the number of input features");
			}

			activeFields = new ArrayList<>(numberOfFeatures);

			for(int i = 0, max = numberOfFeatures; i < max; i++){
				activeFields.add("x" + String.valueOf(i + 1));
			}

			logger.warn("Attribute \'" + ClassDictUtil.formatMember(this, "active_fields") + "\' is not set. Assuming {} as the names of active fields", activeFields);
		}

		List<Feature> result = new ArrayList<>();

		for(String activeField : activeFields){
			DataField dataField = encoder.createDataField(FieldName.create(activeField), opType, dataType);

			result.add(new WildcardFeature(encoder, dataField));
		}

		return result;
	}

	private void encodeOutput(Transformer transformer, Model model, List<OutputField> outputFields){
		SkLearnEncoder encoder = new SkLearnEncoder();

		List<Feature> features = new ArrayList<>();

		Set<FieldName> names = new HashSet<>();

		for(OutputField outputField : outputFields){
			FieldName name = outputField.getName();

			System.out.println(name.getValue());

			DataField dataField = encoder.createDataField(name, outputField.getOpType(), outputField.getDataType());

			features.add(new WildcardFeature(encoder, dataField));

			names.add(name);
		}

		transformer.encodeFeatures(features, encoder);

		class OutputFinder extends AbstractVisitor {

			private Output output = null;


			@Override
			public VisitorAction visit(Output output){

				if(output.hasOutputFields()){
					List<OutputField> outputFields = output.getOutputFields();

					Set<FieldName> definedNames = new HashSet<>();

					for(OutputField outputField : outputFields){
						FieldName name = outputField.getName();

						definedNames.add(name);
					}

					if(definedNames.containsAll(names)){
						setOutput(output);

						return VisitorAction.TERMINATE;
					}
				}

				return super.visit(output);
			}

			public Output getOutput(){
				return this.output;
			}

			private void setOutput(Output output){
				this.output = output;
			}
		}

		OutputFinder outputFinder = new OutputFinder();
		outputFinder.applyTo(model);

		Output output = outputFinder.getOutput();
		if(output == null){
			throw new IllegalArgumentException();
		}

		Map<FieldName, DerivedField> derivedFields = encoder.getDerivedFields();

		for(DerivedField derivedField : derivedFields.values()){
			OutputField outputField = new OutputField(derivedField.getName(), derivedField.getDataType())
				.setOpType(derivedField.getOpType())
				.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
				.setExpression(derivedField.getExpression());

			output.addOutputFields(outputField);
		}
	}

	@Override
	public List<? extends Transformer> getTransformers(){
		List<Object[]> steps = getSteps();

		if(steps.size() > 0){
			steps = steps.subList(0, steps.size() - 1);
		}

		return TupleUtil.extractElementList(steps, 1, Transformer.class);
	}

	@Override
	public Estimator getEstimator(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException("Expected one or more elements, got zero elements");
		}

		Object[] lastStep = steps.get(steps.size() - 1);

		return TupleUtil.extractElement(lastStep, 1, Estimator.class);
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
		Object transformer = get(key);

		if(transformer == null){
			return null;
		}

		return get(key, Transformer.class);
	}

	public List<String> getActiveFields(){

		if(!containsKey("active_fields")){
			return null;
		}

		return (List)getArray("active_fields", String.class);
	}

	public PMMLPipeline setActiveFields(List<String> activeFields){
		put("active_fields", toArray(activeFields));

		return this;
	}

	public List<String> getTargetFields(){

		// SkLearn2PMML 0.24.3
		if(containsKey("target_field")){
			return Collections.singletonList((String)get("target_field"));
		} // End if

		// SkLearn2PMML 0.25+
		if(!containsKey("target_fields")){
			return null;
		}

		return (List)getArray("target_fields", String.class);
	}

	public PMMLPipeline setTargetFields(List<String> targetFields){
		put("target_fields", toArray(targetFields));

		return this;
	}

	public String getRepr(){
		return (String)get("repr_");
	}

	public PMMLPipeline setRepr(String repr){
		put("repr_", repr);

		return this;
	}

	public Verification getVerification(){
		return (Verification)get("verification");
	}

	public PMMLPipeline setVerification(Verification verification){
		put("verification", verification);

		return this;
	}

	static
	private Model getFinalModel(MiningModel miningModel){
		Segmentation segmentation = miningModel.getSegmentation();

		MultipleModelMethod multipleModelMethod = segmentation.getMultipleModelMethod();
		switch(multipleModelMethod){
			case SELECT_FIRST:
			case SELECT_ALL:
				throw new IllegalArgumentException();
			case MODEL_CHAIN:
				{
					List<Segment> segments = segmentation.getSegments();

					Segment lastSegment = segments.get(segments.size() - 1);

					Predicate predicate = lastSegment.getPredicate();
					if(!(predicate instanceof True)){
						throw new IllegalArgumentException();
					}

					Model model = lastSegment.getModel();
					if(model instanceof MiningModel){
						MiningModel finalMiningModel = (MiningModel)model;

						return getFinalModel(finalMiningModel);
					}

					return model;
				}
			default:
				break;
		}

		return miningModel;
	}

	static
	private NDArray toArray(List<String> strings){
		NDArray result = new NDArray();
		result.put("data", strings);
		result.put("fortran_order", Boolean.FALSE);

		return result;
	}

	private static final Logger logger = LoggerFactory.getLogger(PMMLPipeline.class);
}
