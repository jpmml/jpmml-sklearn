/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn_pandas;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import com.google.common.base.CaseFormat;
import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.BayesOutput;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NeuralOutput;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.TransformationDictionary;
import org.dmg.pmml.Value;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.LoggerUtil;
import org.jpmml.sklearn.TupleUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.MultiTransformer;
import sklearn.Transformer;

public class DataFrameMapper extends ClassDict {

	public DataFrameMapper(String module, String name){
		super(module, name);
	}

	public void updatePMML(Schema schema, PMML pmml){
		FieldName targetField = schema.getTargetField();
		List<FieldName> activeFields = schema.getActiveFields();

		List<Object[]> features = new ArrayList<>(getFeatures());

		if(features.size() < 1){
			logger.warn("The list of mappings is empty");

			return;
		} // End if

		if(targetField != null){
			logger.info("Updating 1 target field and {} active field(s)", activeFields.size());

			// Move the target column from the last position to the first position
			features.add(0, features.remove(features.size() - 1));
		} else

		{
			logger.info("Updating {} active field(s)", activeFields.size());
		}

		ListIterator<Object[]> featureIt = features.listIterator();

		DataDictionary dataDictionary = pmml.getDataDictionary();

		List<DataField> dataFields = dataDictionary.getDataFields();

		ListIterator<DataField> dataFieldIt = dataFields.listIterator();

		TransformationDictionary transformationDictionary = pmml.getTransformationDictionary();
		if(transformationDictionary == null){
			transformationDictionary = new TransformationDictionary();

			pmml.setTransformationDictionary(transformationDictionary);
		}

		final
		Map<FieldName, FieldName> updatedTargetFields = new LinkedHashMap<>();

		// The target field
		if(targetField != null){
			Object[] feature = featureIt.next();

			List<FieldName> names = getNameList(feature);
			List<Transformer> transformers = getTransformerList(feature);

			FieldName name = Iterables.getOnlyElement(names);

			if(!transformers.isEmpty()){
				logger.error("Target field {} must not specify a transformation", name);

				throw new IllegalArgumentException();
			}

			DataField dataField = dataFieldIt.next();

			logger.info("Mapping target field {} to {}", dataField.getName(), name);

			updatedTargetFields.put(dataField.getName(), name);

			dataField.setName(name);
		}

		final
		Map<FieldName, Set<FieldName>> updatedActiveFields = new LinkedHashMap<>();

		Set<FieldName> mappedNames = new LinkedHashSet<>();

		// Zero or more active fields
		for(int row = 0; featureIt.hasNext(); row++){
			Object[] feature = featureIt.next();

			final
			List<FieldName> names = getNameList(feature);
			List<Transformer> transformers = getTransformerList(feature);

			Set<FieldName> uniqueNames = new LinkedHashSet<>();
			Set<FieldName> duplicateNames = new LinkedHashSet<>();

			for(FieldName name : names){
				boolean unique = uniqueNames.add(name);

				if(!unique){
					duplicateNames.add(name);
				}
			}

			Sets.SetView<FieldName> duplicateMappedNames = Sets.intersection(uniqueNames, mappedNames);
			if(duplicateMappedNames.size() > 0){
				duplicateMappedNames.copyInto(duplicateNames);
			} // End if

			if(duplicateNames.size() > 0){
				logger.error("Duplicate mappings(s): {}", new ArrayList<>(duplicateNames));

				throw new IllegalArgumentException();
			} // End if

			if(transformers.isEmpty()){
				Transformer transformer = new MultiTransformer(null, null){

					@Override
					public int getNumberOfFeatures(){
						return names.size();
					}

					@Override
					public Expression encode(int index, FieldName name){
						Expression expression = new FieldRef(name);

						return expression;
					}
				};

				transformers = Collections.singletonList(transformer);
			}

			List<FieldName> inputNames = names;

			ListIterator<Transformer> transformerIt = transformers.listIterator();

			for(int column = 0; transformerIt.hasNext(); column++){
				Transformer transformer = transformerIt.next();

				int numberOfInputs = transformer.getNumberOfInputs();
				int numberOfOutputs = transformer.getNumberOfOutputs();

				Step step;

				if(!transformerIt.hasNext()){
					Step finalStep = updateDataDictionary(dataFieldIt, names, transformers.get(0), names.size(), numberOfOutputs);

					logger.info("Mapping active field(s) {} to {}", LoggerUtil.formatNameList(finalStep.getOutputNames()), LoggerUtil.formatNameList(finalStep.getInputNames()));

					for(int i = 0; i < numberOfOutputs; i++){
						FieldName outputName = finalStep.getOutputName(i);

						updatedActiveFields.put(outputName, uniqueNames);
					}

					if(inputNames.size() != numberOfInputs){
						throw new IllegalArgumentException();
					}

					step = new Step(finalStep.getDataType(), finalStep.getOpType(), inputNames, finalStep.getOutputNames());
				} else

				{
					if(inputNames.size() != numberOfInputs){
						throw new IllegalArgumentException();
					}

					List<FieldName> outputNames = new ArrayList<>();

					String className = getClassName(transformer);

					for(int i = 0; i < numberOfOutputs; i++){
						FieldName outputName = FieldName.create(className + "("+ row +"," + column + "," + i + ")");

						outputNames.add(outputName);
					}

					step = new Step(transformer.getDataType(), transformer.getOpType(), inputNames, outputNames);
				}

				for(int i = 0; i < numberOfOutputs; i++){
					FieldName outputName = step.getOutputName(i);

					Expression expression = transformer.encode(i, step.getInputNames());

					DerivedField derivedField = new DerivedField(step.getOpType(), step.getDataType())
						.setName(outputName)
						.setExpression(expression);

					transformationDictionary.addDerivedFields(derivedField);
				}

				inputNames = step.getOutputNames();
			}

			mappedNames.addAll(names);
		}

		if(dataFieldIt.hasNext()){
			Function<DataField, FieldName> function = new Function<DataField, FieldName>(){

				@Override
				public FieldName apply(DataField dataField){
					return dataField.getName();
				}
			};

			List<FieldName> unmappedActiveFields = Lists.newArrayList(Iterators.transform(dataFieldIt, function));

			logger.error("The list of mappings is shorter than the list of fields. Found {} unmapped active field(s): {}", unmappedActiveFields.size(), LoggerUtil.formatNameList(unmappedActiveFields));

			throw new IllegalArgumentException();
		}

		Visitor targetFieldUpdater = new AbstractVisitor(){

			@Override
			public VisitorAction visit(BayesOutput bayesOutput){
				bayesOutput.setFieldName(filterName(bayesOutput.getFieldName()));

				return super.visit(bayesOutput);
			}

			@Override
			public VisitorAction visit(MiningField miningField){
				miningField.setName(filterName(miningField.getName()));

				return super.visit(miningField);
			}

			@Override
			public VisitorAction visit(NeuralOutput neuralOutput){
				DerivedField derivedField = neuralOutput.getDerivedField();

				Expression expression = derivedField.getExpression();

				if(expression instanceof FieldRef){
					FieldRef fieldRef = (FieldRef)expression;

					fieldRef.setField(filterName(fieldRef.getField()));
				} else

				if(expression instanceof NormDiscrete){
					NormDiscrete normDiscrete = (NormDiscrete)expression;

					normDiscrete.setField(filterName(normDiscrete.getField()));
				}

				return super.visit(neuralOutput);
			}

			private FieldName filterName(FieldName name){

				if(updatedTargetFields.containsKey(name)){
					FieldName updatedName = updatedTargetFields.get(name);

					return updatedName;
				}

				return name;
			}
		};

		targetFieldUpdater.applyTo(pmml);

		Visitor activeFieldUpdater = new AbstractVisitor(){

			@Override
			public VisitorAction visit(MiningSchema miningSchema){
				List<MiningField> miningFields = miningSchema.getMiningFields();

				Set<FieldName> names = new LinkedHashSet<>();

				ListIterator<MiningField> miningFieldIt = miningFields.listIterator();
				while(miningFieldIt.hasNext()){
					MiningField miningField = miningFieldIt.next();

					FieldName name = miningField.getName();

					Set<FieldName> updatedNames = updatedActiveFields.get(name);
					if(updatedNames != null){
						names.addAll(updatedNames);

						miningFieldIt.remove();
					}
				}

				for(FieldName name : names){
					MiningField miningField = ModelUtil.createMiningField(name);

					miningFieldIt.add(miningField);
				}

				return super.visit(miningSchema);
			}
		};

		activeFieldUpdater.applyTo(pmml);
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}

	/**
	 * @param transformer The first transformer in the list of transformers.
	 * @param numberOfInputs The number of dimensions in the input feature space.
	 * @param numberOfOutputs The number of dimensions in the output (ie. transformed) feature space.
	 */
	static
	private Step updateDataDictionary(ListIterator<DataField> dataFieldIt, List<FieldName> inputNames, Transformer transformer, int numberOfInputs, int numberOfOutputs){
		OpType opType = null;
		DataType dataType = null;

		if(inputNames.size() != numberOfInputs){
			throw new IllegalArgumentException();
		}

		List<FieldName> outputNames = new ArrayList<>();

		for(int i = 0; i < Math.min(numberOfInputs, numberOfOutputs); i++){
			FieldName inputName = inputNames.get(i);

			DataField dataField = dataFieldIt.next();

			outputNames.add(dataField.getName());

			if(opType != null && !(opType).equals(dataField.getOpType())){
				throw new IllegalArgumentException();
			} // End if

			if(dataType != null && !(dataType).equals(dataField.getDataType())){
				throw new IllegalArgumentException();
			}

			opType = dataField.getOpType();
			dataType = dataField.getDataType();

			updateDataField(dataField, inputName, transformer);
		}

		if(numberOfInputs > numberOfOutputs){
			int count = (numberOfInputs - numberOfOutputs);

			for(int i = 0; i < count; i++){
				FieldName inputName = inputNames.get(numberOfOutputs + i);

				DataField dataField = new DataField();

				dataFieldIt.add(dataField);

				updateDataField(dataField, inputName, transformer);
			}
		} else

		if(numberOfInputs < numberOfOutputs){
			int count = (numberOfOutputs - numberOfInputs);

			for(int i = 0; i < count; i++){
				DataField dataField = dataFieldIt.next();

				outputNames.add(dataField.getName());

				dataFieldIt.remove();
			}
		}

		Step step = new Step(dataType, opType, inputNames, outputNames);

		return step;
	}

	static
	private void updateDataField(DataField dataField, FieldName name, Transformer transformer){
		dataField.setName(name)
			.setOpType(transformer.getOpType())
			.setDataType(transformer.getDataType());

		List<Value> values = dataField.getValues();
		if(values.size() > 0){
			values.clear();
		}

		List<?> classes = transformer.getClasses();

		Function<Object, String> function = new Function<Object, String>(){

			@Override
			public String apply(Object value){
				return ValueUtil.formatValue(value);
			}
		};

		values.addAll(PMMLUtil.createValues(Lists.transform(classes, function)));
	}

	static
	private String getClassName(Transformer transformer){
		Class<? extends Transformer> clazz = transformer.getClass();

		String name = clazz.getSimpleName();

		return CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, name);
	}

	static
	private List<FieldName> getNameList(Object[] feature){
		Function<Object, FieldName> function = new Function<Object, FieldName>(){

			@Override
			public FieldName apply(Object object){

				if(object instanceof String){
					return FieldName.create((String)object);
				}

				throw new IllegalArgumentException("The key object (" + ClassDictUtil.formatClass(object) + ") is not a String");
			}
		};

		try {
			if(feature[0] instanceof List){
				return new ArrayList<>(Lists.transform(((List)feature[0]), function));
			}

			return Collections.singletonList(function.apply(feature[0]));
		} catch(RuntimeException re){
			throw new IllegalArgumentException("Invalid mapping key", re);
		}
	}

	static
	private List<Transformer> getTransformerList(Object[] feature){
		Function<Object, Transformer> function = new Function<Object, Transformer>(){

			@Override
			public Transformer apply(Object object){

				if(object instanceof Transformer){
					return (Transformer)object;
				}

				throw new IllegalArgumentException("The value object (" + ClassDictUtil.formatClass(object) + ") is not a Transformer or is not a supported Transformer subclass");
			}
		};

		try {
			if(feature[1] == null){
				return Collections.emptyList();
			} // End if

			if(feature[1] instanceof TransformerPipeline){
				TransformerPipeline transformerPipeline = (TransformerPipeline)feature[1];

				List<Object[]> steps = transformerPipeline.getSteps();

				return new ArrayList<>(Lists.transform((List)TupleUtil.extractElement(steps, 1), function));
			} // End if

			if(feature[1] instanceof List){
				return new ArrayList<>(Lists.transform((List)feature[1], function));
			}

			return Collections.singletonList(function.apply(feature[1]));
		} catch(RuntimeException re){
			throw new IllegalArgumentException("Invalid mapping value", re);
		}
	}

	static
	private class Step {

		private DataType dataType = null;

		private OpType opType = null;

		private List<FieldName> inputNames = null;

		private List<FieldName> outputNames = null;


		private Step(DataType dataType, OpType opType, List<FieldName> inputNames, List<FieldName> outputNames){
			setDataType(dataType);
			setOpType(opType);
			setInputNames(inputNames);
			setOutputNames(outputNames);
		}

		public DataType getDataType(){
			return this.dataType;
		}

		private void setDataType(DataType dataType){
			this.dataType = dataType;
		}

		public OpType getOpType(){
			return this.opType;
		}

		private void setOpType(OpType opType){
			this.opType = opType;
		}

		public FieldName getInputName(int index){
			List<FieldName> inputNames = getInputNames();

			return inputNames.get(index);
		}

		public List<FieldName> getInputNames(){
			return this.inputNames;
		}

		private void setInputNames(List<FieldName> inputNames){
			this.inputNames = inputNames;
		}

		public FieldName getOutputName(int index){
			List<FieldName> outputNames = getOutputNames();

			return outputNames.get(index);
		}

		public List<FieldName> getOutputNames(){
			return this.outputNames;
		}

		private void setOutputNames(List<FieldName> outputNames){
			this.outputNames = outputNames;
		}
	}

	private static final Logger logger = LoggerFactory.getLogger(DataFrameMapper.class);
}