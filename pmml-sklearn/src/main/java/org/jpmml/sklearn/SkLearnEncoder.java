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

import numpy.DType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.model.ReflectionUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.PythonEncoder;
import sklearn.Classifier;
import sklearn.Estimator;
import sklearn.ScalarLabelUtil;
import sklearn.ensemble.hist_gradient_boosting.TreePredictor;
import sklearn.neighbors.BinaryTree;
import sklearn.tree.Tree;
import sklearn2pmml.decoration.Alias;
import sklearn2pmml.decoration.Domain;

public class SkLearnEncoder extends PythonEncoder {

	private Map<String, Domain> domains = new LinkedHashMap<>();

	private Model model = null;


	public SkLearnEncoder(){
	}

	public List<Feature> export(Model model, String name){
		return export(model, Collections.singletonList(name));
	}

	public List<Feature> export(Model model, List<String> names){
		Output output = model.getOutput();

		List<OutputField> outputFields = output.getOutputFields();

		List<Feature> result = new ArrayList<>();

		for(String name : names){
			DerivedOutputField derivedOutputField = null;

			List<OutputField> nameOutputFields = selectOutputFields(name, outputFields);
			for(OutputField nameOutputField : nameOutputFields){
				derivedOutputField = createDerivedField(model, nameOutputField, true);
			}

			Feature feature;

			OpType opType = derivedOutputField.getOpType();
			switch(opType){
				case CATEGORICAL:
					feature = new CategoricalFeature(this, derivedOutputField);
					break;
				case CONTINUOUS:
					feature = new ContinuousFeature(this, derivedOutputField);
					break;
				default:
					throw new IllegalArgumentException();
			}

			result.add(feature);

			outputFields.removeAll(nameOutputFields);
		}

		return result;
	}

	public Feature exportPrediction(Model model, ScalarLabel scalarLabel){
		return exportPrediction(model, FieldNameUtil.create(Estimator.FIELD_PREDICT, scalarLabel.getName()), scalarLabel);
	}

	public Feature exportPrediction(Model model, String name, ScalarLabel scalarLabel){
		OutputField outputField = ModelUtil.createPredictedField(name, ScalarLabelUtil.getOpType(scalarLabel), scalarLabel.getDataType())
			.setFinalResult(false);

		DerivedOutputField derivedOutputField = createDerivedField(model, outputField, false);

		return ScalarLabelUtil.toFeature(scalarLabel, derivedOutputField, this);
	}

	public Feature exportProbability(Model model, Object value){
		return exportProbability(model, FieldNameUtil.create(Classifier.FIELD_PROBABILITY, value), value);
	}

	public Feature exportProbability(Model model, String name, Object value){
		OutputField probabilityOutputField = ModelUtil.createProbabilityField(name, DataType.DOUBLE, value)
			.setFinalResult(false);

		DerivedOutputField probabilityField = createDerivedField(model, probabilityOutputField, false);

		return new ContinuousFeature(this, probabilityField);
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
		} catch(RuntimeException re){
			String name = derivedField.requireName();

			String message = "Field " + name + " is already defined. " +
				"Please refactor the pipeline so that it would not contain duplicate field declarations, " +
				"or use the " + (Alias.class).getName() + " wrapper class to override the default name with a custom name (eg. " + Alias.formatAliasExample() + ")";

			throw new IllegalArgumentException(message, re);
		}
	}

	public void renameFeature(Feature feature, String renamedName){
		String name = feature.getName();

		org.dmg.pmml.Field<?> pmmlField = getField(name);

		if(pmmlField instanceof DataField){
			throw new IllegalArgumentException("User input field " + name + " cannot be renamed");
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
		return this.domains.containsKey(name);
	}

	public Domain getDomain(String name){
		return this.domains.get(name);
	}

	public void setDomain(String name, Domain domain){

		if(domain != null){
			this.domains.put(name, domain);
		} else

		{
			this.domains.remove(name);
		}
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
		DType.addDefinition(Tree.DTYPE_TREE);
		DType.addDefinition(TreePredictor.DTYPE_PREDICTOR_OLD);
		DType.addDefinition(TreePredictor.DTYPE_PREDICTOR_NEW);
	}
}