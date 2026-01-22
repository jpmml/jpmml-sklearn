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
import java.util.Comparator;
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
import org.dmg.pmml.Field;
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
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.NamingException;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaException;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.Attribute;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.SkLearnException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.HasClasses;
import sklearn.Step;
import sklearn.Transformer;
import sklearn.pipeline.SkLearnPipeline;
import sklearn2pmml.Customization;
import sklearn2pmml.CustomizationUtil;
import sklearn2pmml.HasPMMLOptions;
import sklearn2pmml.decoration.Domain;

public class PMMLPipeline extends SkLearnPipeline implements HasPMMLOptions<PMMLPipeline> {

	public PMMLPipeline(){
		this("sklearn2pmml.pipeline", "PMMLPipeline");
	}

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	@Override
	public PMML encodePMML(){
		SkLearnEncoder encoder = new SkLearnEncoder();

		List<String> activeFields = getActiveFields();
		List<String> probabilityFields = null;
		List<String> targetFields = getTargetFields();

		Map<String, ?> header = getHeader();
		String repr = getRepr();
		Transformer predictTransformer = getPredictTransformer();
		Transformer predictProbaTransformer = getPredictProbaTransformer();
		Transformer applyTransformer = getApplyTransformer();
		Verification verification = getVerification();

		List<Customization> customizations = null;

		Estimator estimator = null;

		if(hasFinalEstimator()){
			estimator = getFinalEstimator();

			targetFields = initLabel(targetFields, encoder);

			customizations = estimator.getPMMLCustomizations();
		}

		activeFields = initFeatures(activeFields, encoder);

		if(estimator == null){
			return encodePMML(header, null, repr, encoder);
		}

		Schema schema = encoder.createSchema();

		Model model = estimator.encode(schema);

		encoder.setModel(model);

		if(!estimator.hasFeatureImportances()){
			List<Number> featureImportances = getPMMLFeatureImportances();

			if(featureImportances != null){
				ClassDictUtil.checkSize(activeFields, featureImportances);

				for(int i = 0; i < activeFields.size(); i++){
					String activeField = activeFields.get(i);
					Number featureImportance = featureImportances.get(i);

					DataField dataField = encoder.getDataField(activeField);
					if(dataField == null){
						throw new SchemaException("Field " + ExceptionUtil.formatName(activeField) + " is not defined");
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
				Label label = schema.getLabel();

				List<OutputField> predictFields = new ArrayList<>();

				List<ScalarLabel> scalarLabels = ScalarLabelUtil.toScalarLabels(label);
				for(ScalarLabel scalarLabel : scalarLabels){
					OutputField predictField = ModelUtil.createPredictedField(FieldNameUtil.create(Estimator.FIELD_PREDICT, scalarLabel.getName()), scalarLabel.getOpType(), scalarLabel.getDataType())
						.setFinalResult(false);

					// Disambiguate target fields
					if(label instanceof MultiLabel){
						MultiLabel multiLabel = (MultiLabel)label;

						predictField.setTargetField(scalarLabel.getName());
					}

					encoder.createDerivedField(model, predictField, false);

					predictFields.add(predictField);
				}

				encodeOutput(output, predictFields, predictTransformer, encoder);
			} // End if

			if(predictProbaTransformer != null){
				CategoricalLabel categoricalLabel = schema.requireCategoricalLabel();

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
				Attribute attribute = new Attribute(this, "verify(X)");

				logger.warn("Model verification data is not set. Use the " + ExceptionUtil.formatName(attribute.format()) + " method to correct this deficiency");

				break verification;
			}

			Label label = schema.getLabel();

			List<?> activeValues = verification.getActiveValues();
			int[] activeValuesShape = verification.getActiveValuesShape();

			ClassDictUtil.checkShapes(1, activeFields.size(), activeValuesShape);

			int rows = activeValuesShape[0];

			Map<VerificationField, List<?>> data = new LinkedHashMap<>();

			for(int i = 0; i < activeFields.size(); i++){
				VerificationField verificationField = ModelUtil.createVerificationField(activeFields.get(i));

				Domain domain = encoder.getDomain(verificationField.requireField());

				data.put(verificationField, CMatrixUtil.getColumn(cleanValues(domain, activeValues), rows, activeFields.size(), i));
			}

			Number precision = verification.getPrecision();
			Number zeroThreshold = verification.getZeroThreshold();

			List<ScalarLabel> scalarLabels = ScalarLabelUtil.toScalarLabels(label);

			boolean hasProbabilityValues = verification.hasProbabilityValues();

			if(estimator instanceof HasClasses){
				HasClasses hasClasses = (HasClasses)estimator;

				hasProbabilityValues &= hasClasses.hasProbabilityDistribution();
			} // End if

			if(hasProbabilityValues){
				List<Number> probabilityValues = verification.getProbabilityValues();
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
		} // End if

		if(customizations != null && !customizations.isEmpty()){

			try {
				CustomizationUtil.customize(model, customizations);
			} catch(Exception e){
				throw new SkLearnException("Failed to customize PMML model", e);
			}
		}

		return encodePMML(header, model, repr, encoder);
	}

	private PMML encodePMML(Map<String, ?> header, Model model, String repr, SkLearnEncoder encoder){
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
		SkLearnEncoder outputEncoder = new SkLearnEncoder(){

			@Override
			public void addTransformer(Model transformer){
				throw new UnsupportedOperationException();
			}

			@Override
			public void addDataField(DataField dataField){
				checkField(dataField.requireName());

				super.addDataField(dataField);
			}

			@Override
			public void addDerivedField(DerivedField derivedField){
				checkField(derivedField.requireName());

				super.addDerivedField(derivedField);
			}

			@Override
			public Field<?> getField(String name){

				try {
					return super.getField(name);
				} catch(SchemaException se){
					return encoder.getField(name);
				}
			}

			@Override
			public boolean isFrozen(String name){
				return true;
			}

			@Override
			public Map<String, Domain> getDomains(){
				throw new UnsupportedOperationException();
			}

			@Override
			public Map<String, Feature> getMemory(){
				return encoder.getMemory();
			}

			private void checkField(String name){
				Field<?> field;

				try {
					field = encoder.getField(name);
				} catch(SchemaException se){
					return;
				}

				if(field instanceof DerivedOutputField){
					DerivedOutputField derivedOutputField = (DerivedOutputField)field;

					return;
				}

				throw new NamingException("Field " + ExceptionUtil.formatName(name) + " is already defined");
			}
		};

		Model model = encoder.getModel();
		if(model != null){
			outputEncoder.setModel(model);
		}

		List<Feature> features = new ArrayList<>();

		for(OutputField outputField : outputFields){
			DataField dataField = outputEncoder.createDataField(outputField.requireName(), outputField.requireOpType(), outputField.requireDataType());

			features.add(new WildcardFeature(outputEncoder, dataField));
		}

		List<Feature> outputFeatures = transformer.encode(features, outputEncoder);

		Map<String, Integer> finalResults = new LinkedHashMap<>();

		for(Feature outputFeature : outputFeatures){
			String name = outputFeature.getName();

			finalResults.put(name, finalResults.size());
		}

		Collection<DerivedField> derivedFields = (outputEncoder.getDerivedFields()).values();

		for(Iterator<DerivedField> it = derivedFields.iterator(); it.hasNext(); ){
			DerivedField derivedField = it.next();

			OutputField outputField;

			if(derivedField instanceof DerivedOutputField){
				DerivedOutputField derivedOutputField = (DerivedOutputField)derivedField;

				outputField = derivedOutputField.getOutputField();
			} else

			{
				String name = derivedField.requireName();

				outputField = new OutputField(name, derivedField.requireOpType(), derivedField.requireDataType())
					.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
					.setFinalResult(finalResults.containsKey(name))
					.setExpression(derivedField.requireExpression());
			}

			output.addOutputFields(outputField);
		}

		Comparator<OutputField> comparator = new Comparator<OutputField>(){

			@Override
			public int compare(OutputField left, OutputField right){
				int leftIndex = finalResults.getOrDefault(left.requireName(), -1);
				int rightIndex = finalResults.getOrDefault(right.requireName(), -1);

				return Integer.compare(leftIndex, rightIndex);
			}
		};

		Collections.sort(output.getOutputFields(), comparator);

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

	@Override
	public Map<String, ?> getPMMLOptions(){

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			return estimator.getPMMLOptions();
		}

		return null;
	}

	@Override
	public PMMLPipeline setPMMLOptions(Map<String, ?> pmmlOptions){

		if(hasFinalEstimator()){
			Estimator estimator = getFinalEstimator();

			estimator.setPMMLOptions(pmmlOptions);
		}

		return this;
	}

	public Map<String, ?> getHeader(){
		return getOptionalDict("header");
	}

	public List<Number> getPMMLFeatureImportances(){

		if(!hasattr("pmml_feature_importances_")){
			return null;
		}

		return getNumberArray("pmml_feature_importances_");
	}

	public Transformer getPredictTransformer(){
		return getOptionalTransformer("predict_transformer");
	}

	public Transformer getPredictProbaTransformer(){
		return getOptionalTransformer("predict_proba_transformer");
	}

	public Transformer getApplyTransformer(){
		return getOptionalTransformer("apply_transformer");
	}

	public List<String> getActiveFields(){

		if(!hasattr("active_fields")){
			return null;
		}

		return getStringListLike("active_fields");
	}

	public PMMLPipeline setActiveFields(List<String> activeFields){
		setattr("active_fields", NDArrayUtil.toArray(activeFields));

		return this;
	}

	public List<String> getTargetFields(){

		// SkLearn2PMML 0.24.3
		if(hasattr("target_field")){
			return Collections.singletonList(getOptionalString("target_field"));
		} // End if

		// SkLearn2PMML 0.25+
		if(!hasattr("target_fields")){
			return null;
		}

		return getStringListLike("target_fields");
	}

	public PMMLPipeline setTargetFields(List<String> targetFields){
		setattr("target_fields", NDArrayUtil.toArray(targetFields));

		return this;
	}

	public String getRepr(){
		return getOptionalString("repr_");
	}

	public PMMLPipeline setRepr(String repr){
		setattr("repr_", repr);

		return this;
	}

	public Verification getVerification(){
		return getOptional("verification", Verification.class);
	}

	public PMMLPipeline setVerification(Verification verification){
		setattr("verification", verification);

		return this;
	}

	@Override
	protected List<String> initTargetFields(Estimator estimator){
		List<String> targetFields = super.initTargetFields(estimator);

		Attribute attribute = new Attribute(this, "target_fields");

		logger.warn("Attribute " + ExceptionUtil.formatName(attribute.format()) + " is not set. Assuming {} as the name(s) of the target field(s)", targetFields);

		return targetFields;
	}

	@Override
	protected List<String> initActiveFields(Step step){
		List<String> activeFields = super.initActiveFields(step);

		Attribute attribute = new Attribute(this, "active_fields");

		logger.warn("Attribute " + ExceptionUtil.formatName(attribute.format()) + " is not set. Assuming {} as the names of active fields", activeFields);

		return activeFields;
	}

	private List<String> initProbabilityFields(CategoricalLabel categoricalLabel){
		List<String> probabilityFields = new ArrayList<>();

		List<?> values = categoricalLabel.getValues();
		for(Object value : values){
			probabilityFields.add(FieldNameUtil.create(Classifier.FIELD_PROBABILITY, value));
		}

		return probabilityFields;
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