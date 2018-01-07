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
package sklearn2pmml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.parsers.DocumentBuilder;

import numpy.core.NDArray;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Extension;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.MiningBuildTask;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelVerification;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Row;
import org.dmg.pmml.VerificationField;
import org.dmg.pmml.VerificationFields;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.PyClassDict;
import org.jpmml.sklearn.SkLearnEncoder;
import org.jpmml.sklearn.TupleUtil;
import org.jpmml.sklearn.XMLUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import sklearn.Classifier;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
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

						DataType dataType = TypeUtil.getDataType(classes, DataType.STRING);

						List<String> categories = ClassifierUtil.formatTargetCategories(classes);

						DataField dataField = encoder.createDataField(FieldName.create(targetField), OpType.CATEGORICAL, dataType, categories);

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

		Transformer transformer = TransformerUtil.getHead(transformers);
		if(transformer != null){

			if(!(transformer instanceof Initializer)){
				features = initFeatures(transformer, transformer.getOpType(), transformer.getDataType(), encoder);
			}

			features = encodeFeatures(features, encoder);
		} else

		{
			features = initFeatures(estimator, estimator.getOpType(), estimator.getDataType(), encoder);
		}

		int numberOfFeatures = estimator.getNumberOfFeatures();
		if(numberOfFeatures > -1){
			ClassDictUtil.checkSize(numberOfFeatures, features);
		}

		Schema schema = new Schema(label, features);

		Model model = estimator.encodeModel(schema);

		if(estimator.isSupervised() && verification != null){

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

			if(estimator instanceof Classifier){
				Classifier classifier = (Classifier)estimator;

				if(classifier.hasProbabilityDistribution() && verification.hasProbabilityValues()){
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
			}

			Number precision = verification.getPrecision();
			Number zeroThreshold = verification.getZeroThreshold();

			int rows = activeValuesShape[0];

			VerificationFields verificationFields = new VerificationFields();

			List<List<?>> data = new ArrayList<>();

			if(activeFields != null){

				for(int i = 0; i < activeFields.size(); i++){
					VerificationField verificationField = createVerificationField(activeFields.get(i));

					verificationFields.addVerificationFields(verificationField);

					data.add(CMatrixUtil.getColumn(activeValues, rows, activeFields.size(), i));
				}
			} // End if

			if(probabilityFields != null){

				for(int i = 0; i < probabilityFields.size(); i++){
					VerificationField verificationField = createVerificationField(probabilityFields.get(i))
						.setPrecision(precision.doubleValue())
						.setZeroThreshold(zeroThreshold.doubleValue());

					verificationFields.addVerificationFields(verificationField);

					data.add(CMatrixUtil.getColumn(probabilityValues, rows, probabilityFields.size(), i));
				}
			} else

			{
				for(int i = 0; i < targetFields.size(); i++){
					VerificationField verificationField = createVerificationField(targetFields.get(i));

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

					verificationFields.addVerificationFields(verificationField);

					data.add(CMatrixUtil.getColumn(targetValues, rows, targetFields.size(), i));
				}
			}

			List<String> keys = new ArrayList<>();

			for(VerificationField verificationField : verificationFields){
				keys.add(verificationField.getColumn());
			}

			DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

			InlineTable inlineTable = new InlineTable();

			for(int i = 0; i < rows; i++){
				Row row = new Row();

				Document document = documentBuilder.newDocument();

				for(int j = 0; j < data.size(); j++){
					List<?> column = data.get(j);

					Object cell = column.get(i);
					if(cell == null){
						continue;
					}

					Element element = document.createElement(keys.get(j));
					element.setTextContent(ValueUtil.formatValue(cell));

					row.addContent(element);
				}

				inlineTable.addRows(row);
			}

			ModelVerification modelVerification = new ModelVerification(verificationFields, inlineTable)
				.setRecordCount(rows);

			model.setModelVerification(modelVerification);
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
	private VerificationField createVerificationField(String name){
		String tagName = name;

		Matcher matcher = PMMLPipeline.FUNCTION.matcher(tagName);
		if(matcher.matches()){
			tagName = (matcher.group(1) + "_" + matcher.group(2));
		}

		VerificationField verificationField = new VerificationField()
			.setField(FieldName.create(name))
			.setColumn(XMLUtil.createTagName(tagName));

		return verificationField;
	}

	static
	private NDArray toArray(List<String> strings){
		NDArray result = new NDArray();
		result.put("data", strings);
		result.put("fortran_order", Boolean.FALSE);

		return result;
	}

	private static final Pattern FUNCTION = Pattern.compile("^(.+)\\((.+)\\)$");

	private static final Logger logger = LoggerFactory.getLogger(PMMLPipeline.class);
}
