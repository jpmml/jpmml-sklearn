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
import java.util.LinkedHashMap;
import java.util.Map;

import numpy.DType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.PythonEncoder;
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
			Field field = (Feature.class).getDeclaredField("name");

			if(!field.isAccessible()){
				field.setAccessible(true);
			}

			field.set(feature, renamedName);
		} catch(ReflectiveOperationException roe){
			throw new RuntimeException(roe);
		}

		derivedField.setName(renamedName);

		addDerivedField(derivedField);
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

	static {
		PickleUtil.init("sklearn2pmml.properties");

		DType.addDefinition(BinaryTree.DTYPE_NODEDATA);
		DType.addDefinition(Tree.DTYPE_TREE);
		DType.addDefinition(TreePredictor.DTYPE_PREDICTOR_OLD);
		DType.addDefinition(TreePredictor.DTYPE_PREDICTOR_NEW);
	}
}