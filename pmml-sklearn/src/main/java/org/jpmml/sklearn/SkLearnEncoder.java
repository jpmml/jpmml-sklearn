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
package org.jpmml.sklearn;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import numpy.DType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.NamingException;
import org.jpmml.converter.ResolutionException;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.model.ReflectionUtil;
import org.jpmml.model.UnsupportedAttributeException;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.PythonEncoder;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasMultiType;
import sklearn.Step;
import sklearn.StepUtil;
import sklearn.ensemble.hist_gradient_boosting.TreePredictor;
import sklearn.neighbors.BinaryTree;
import sklearn.tree.Tree;
import sklearn2pmml.decoration.Alias;
import sklearn2pmml.decoration.Domain;

public class SkLearnEncoder extends PythonEncoder {

	private Map<String, Domain> domains = new LinkedHashMap<>();

	private Label label = null;

	private List<? extends Feature> features = Collections.emptyList();

	private Map<String, Feature> memory = new LinkedHashMap<>();

	private Predicate predicate = null;

	private Model model = null;


	public SkLearnEncoder(){
	}

	@Override
	public void addTransformer(Model transformer){

		if(hasModel()){
			throw new IllegalStateException("Model is already defined");
		}

		super.addTransformer(transformer);
	}

	@Override
	public Model encodeModel(Model model){
		Predicate predicate = getPredicate();

		model = super.encodeModel(model);

		if(predicate == null){
			return model;
		}

		MiningModel miningModel = (MiningModel)model;

		Segmentation segmentation = miningModel.requireSegmentation();

		Segmentation.MultipleModelMethod multipleModelMethod = segmentation.requireMultipleModelMethod();
		switch(multipleModelMethod){
			case MODEL_CHAIN:
				break;
			default:
				throw new UnsupportedAttributeException(segmentation, multipleModelMethod);
		}

		List<Segment> segments = segmentation.requireSegments();

		Segment finalSegment = segments.get(segments.size() - 1);

		finalSegment.setPredicate(predicate);

		Set<MiningFunction> miningFunctions = segments.stream()
			.map(segment -> {
				Model segmentModel = segment.requireModel();

				return segmentModel.requireMiningFunction();
			})
			.collect(Collectors.toSet());

		if(miningFunctions.size() > 1){
			miningModel.setMiningFunction(MiningFunction.MIXED);
		}

		return miningModel;
	}

	public Label initLabel(Estimator estimator, List<String> names){
		List<? extends Feature> features = getFeatures();

		if(!features.isEmpty()){
			throw new IllegalStateException();
		}

		Label label = estimator.encodeLabel(names, this);

		setLabel(label);

		return label;
	}

	public List<Feature> initFeatures(Step step, List<String> names){
		HasMultiType hasMultiType = StepUtil.getType(step);

		List<Feature> features = new ArrayList<>();

		for(int i = 0; i < names.size(); i++){
			String name = names.get(i);

			OpType opType = hasMultiType.getOpType(i);
			DataType dataType = hasMultiType.getDataType(i);

			DataField dataField = createDataField(name, opType, dataType);

			Feature feature = new WildcardFeature(this, dataField);

			features.add(feature);
		}

		setFeatures(features);

		return features;
	}

	public Schema createSchema(){
		Label label = getLabel();
		List<? extends Feature> features = getFeatures();

		return new Schema(this, label, features);
	}

	public List<Feature> export(Model model, String name){
		return export(model, Collections.singletonList(name));
	}

	public List<Feature> export(Model model, List<String> names){
		Output output = EstimatorUtil.getFinalOutput(model);
		if(output == null){
			throw new IllegalArgumentException();
		}

		List<OutputField> outputFields = output.getOutputFields();

		List<Feature> result = new ArrayList<>();

		for(String name : names){
			DerivedOutputField derivedOutputField = null;

			List<OutputField> nameOutputFields = selectOutputFields(name, outputFields);
			for(OutputField nameOutputField : nameOutputFields){
				derivedOutputField = createDerivedField(model, nameOutputField, true);
			}

			Feature feature = derivedOutputField.toFeature(this);

			result.add(feature);

			outputFields.removeAll(nameOutputFields);
		}

		return result;
	}

	public Feature exportPrediction(Model model, ScalarLabel scalarLabel){
		String name;

		if(scalarLabel.isAnonymous()){
			name = Estimator.FIELD_PREDICT;
		} else

		{
			name = FieldNameUtil.create(Estimator.FIELD_PREDICT, scalarLabel.getName());
		}

		return exportPrediction(model, name, scalarLabel);
	}

	public Feature exportPrediction(Model model, String name, ScalarLabel scalarLabel){
		OutputField outputField = ModelUtil.createPredictedField(name, scalarLabel.getOpType(), scalarLabel.getDataType())
			.setFinalResult(false);

		DerivedOutputField derivedOutputField = createDerivedField(model, outputField, false);

		return derivedOutputField.toFeature(this);
	}

	public Feature exportProbability(Model model, Object value){
		return exportProbability(model, FieldNameUtil.create(Classifier.FIELD_PROBABILITY, value), value);
	}

	public Feature exportProbability(Model model, String name, Object value){
		OutputField probabilityOutputField = ModelUtil.createProbabilityField(name, DataType.DOUBLE, value)
			.setFinalResult(false);

		DerivedOutputField probabilityField = createDerivedField(model, probabilityOutputField, false);

		return probabilityField.toFeature(this);
	}

	public DataField createDataField(String name){
		return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
	}

	public DerivedField createDerivedField(String name, Expression expression){
		return createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);
	}

	@Override
	public void addDerivedField(DerivedField derivedField){

		try {
			super.addDerivedField(derivedField);
		} catch(NamingException ne){
			String name = derivedField.requireName();

			String message = "Field " + name + " is already defined";
			String solution =
				"Refactor the pipeline so that it would not contain duplicate field declarations, " +
				"or use the " + (Alias.class).getName() + " wrapper class to override the default name with a custom name (eg. " + Alias.formatAliasExample() + ")";

			throw new SkLearnException(message, ne)
				.setSolution(solution);
		}
	}

	public void renameFeature(Feature feature, String renamedName){
		String name = feature.getName();

		if(Objects.equals(name, renamedName)){
			return;
		}

		org.dmg.pmml.Field<?> pmmlField = getField(name);

		if(pmmlField instanceof DataField){
			throw new SkLearnException("Field " + name + " cannot be renamed")
				.setSolution("Rename input fields in Python beforehand (eg. as DataFrame columns)");
		}

		org.dmg.pmml.Field renamedPmmlField;

		try {
			renamedPmmlField = getField(renamedName);

			throw new SkLearnException("Field " + renamedName + " is already defined")
				.setSolution("Choose a different name");
		} catch(ResolutionException re){
			// Ignored
		}

		DerivedField derivedField = removeDerivedField(name);

		try {
			Field nameField = (Feature.class).getDeclaredField("name");

			ReflectionUtil.setFieldValue(nameField, feature, renamedName);
		} catch(ReflectiveOperationException roe){
			throw new RuntimeException(roe);
		}

		derivedField.setName(renamedName);

		addDerivedField(derivedField);
	}

	public void renameFeatures(List<Feature> features, List<String> renamedNames){
		ClassDictUtil.checkSize(renamedNames.size(), features);

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			String renamedName = renamedNames.get(i);

			renameFeature(feature, renamedName);
		}
	}

	public boolean isFrozen(String name){
		Map<String, Domain> domains = getDomains();

		return domains.containsKey(name);
	}

	public Domain getDomain(String name){
		Map<String, Domain> domains = getDomains();

		return domains.get(name);
	}

	public void setDomain(String name, Domain domain){
		Map<String, Domain> domains = getDomains();

		if(domain != null){
			domains.put(name, domain);
		} else

		{
			domains.remove(name);
		}
	}

	public Map<String, Domain> getDomains(){
		return this.domains;
	}

	public Label getLabel(){
		return this.label;
	}

	public void setLabel(Label label){
		this.label = label;
	}

	public List<? extends Feature> getFeatures(){
		return this.features;
	}

	public void setFeatures(List<? extends Feature> features){
		this.features = Objects.requireNonNull(features);
	}

	public void memorize(String name, Feature feature){
		Map<String, Feature> memory = getMemory();

		memory.put(name, feature);
	}

	public Feature recall(String name){
		Map<String, Feature> memory = getMemory();

		return memory.get(name);
	}

	public Map<String, Feature> getMemory(){
		return this.memory;
	}

	public Predicate getPredicate(){
		return this.predicate;
	}

	public void setPredicate(Predicate predicate){
		this.predicate = predicate;
	}

	public boolean hasModel(){
		Model model = getModel();

		return (model != null);
	}

	public Model getModel(){
		return this.model;
	}

	public void setModel(Model model){
		this.model = model;
	}

	static
	public boolean isPrediction(OutputField outputField){
		ResultFeature resultFeature = outputField.getResultFeature();

		switch(resultFeature){
			case PREDICTED_VALUE:
			case TRANSFORMED_VALUE:
			case DECISION:
				return true;
			default:
				return false;
		}
	}

	static
	private List<OutputField> selectOutputFields(String name, List<OutputField> outputFields){
		List<OutputField> result = new ArrayList<>();

		for(OutputField outputField : outputFields){
			boolean prediction = isPrediction(outputField);

			if(prediction){
				result.add(outputField);
			} // End if

			if(!Objects.equals(name, outputField.requireName())){
				continue;
			} // End if

			if(prediction){
				return result;
			} else

			{
				return Collections.singletonList(outputField);
			}
		}

		throw new IllegalArgumentException(name);
	}

	static {
		ClassLoader clazzLoader = SkLearnEncoder.class.getClassLoader();

		PickleUtil.init(clazzLoader, "sklearn2pmml.properties");

		DType.addDefinition(BinaryTree.DTYPE_NODEDATA);
		DType.addDefinition(Tree.DTYPE_TREE_OLD);
		DType.addDefinition(Tree.DTYPE_TREE_NEW);
		DType.addDefinition(TreePredictor.DTYPE_PREDICTOR_OLD);
		DType.addDefinition(TreePredictor.DTYPE_PREDICTOR_NEW);
	}
}