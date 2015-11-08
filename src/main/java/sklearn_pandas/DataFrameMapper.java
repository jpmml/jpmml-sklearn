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
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.SetMultimap;
import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.BayesOutput;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import org.jpmml.sklearn.SchemaUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.ComplexTransformer;
import sklearn.SimpleTransformer;
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

		Model model = Iterables.getOnlyElement(pmml.getModels());

		LocalTransformations localTransformations = model.getLocalTransformations();
		if(localTransformations == null){
			localTransformations = new LocalTransformations();

			model.setLocalTransformations(localTransformations);
		}

		final
		Map<FieldName, FieldName> updatedTargetFields = new LinkedHashMap<>();

		final
		SetMultimap<FieldName, FieldName> updatedActiveFields = LinkedHashMultimap.create();

		// The target field
		if(targetField != null){
			Object[] feature = featureIt.next();

			FieldName name = getField(feature);
			Transformer transformer = getTransformer(feature);

			if(transformer != null){
				logger.error("Target field {} must not specify a transformation", name);

				throw new IllegalArgumentException();
			}

			DataField dataField = dataFieldIt.next();

			logger.info("Mapping target field {} to {}", dataField.getName(), name);

			updatedTargetFields.put(dataField.getName(), name);

			dataField.setName(name);
		}

		// Zero or more active fields
		while(featureIt.hasNext()){
			Object[] feature = featureIt.next();

			Transformer transformer = getTransformer(feature);

			if(transformer == null){
				transformer = new SimpleTransformer(null, null){

					@Override
					public Expression encode(FieldName name){
						FieldRef fieldRef = new FieldRef(name);

						return fieldRef;
					}
				};
			} // End if

			if(transformer instanceof SimpleTransformer){
				SimpleTransformer simpleTransformer = (SimpleTransformer)transformer;

				FieldName name = getField(feature);

				DataField dataField = dataFieldIt.next();

				logger.info("Mapping active field {} to {}", dataField.getName(), name);

				updatedActiveFields.put(dataField.getName(), name);

				Expression expression = simpleTransformer.encode(name);

				DerivedField derivedField = new DerivedField(dataField.getOpType(), dataField.getDataType())
					.setName(dataField.getName())
					.setExpression(expression);

				localTransformations.addDerivedFields(derivedField);

				updateDataField(dataField, name, simpleTransformer);
			} else

			if(transformer instanceof ComplexTransformer){
				ComplexTransformer complexTransformer = (ComplexTransformer)transformer;

				int numberOfInputs = complexTransformer.getNumberOfInputs();
				int numberOfOutputs = complexTransformer.getNumberOfOutputs();

				List<FieldName> inputNames = getFieldList(feature, numberOfInputs);
				List<FieldName> outputNames = new ArrayList<>();

				OpType opType = null;
				DataType dataType = null;

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

					updateDataField(dataField, inputName, complexTransformer);
				}

				if(numberOfInputs > numberOfOutputs){
					int count = (numberOfInputs - numberOfOutputs);

					for(int i = 0; i < count; i++){
						FieldName inputName = inputNames.get(numberOfOutputs + i);

						DataField dataField = new DataField();

						dataFieldIt.add(dataField);

						updateDataField(dataField, inputName, complexTransformer);
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

				logger.info("Mapping active field(s) {} to {}", outputNames, inputNames);

				for(int i = 0; i < numberOfOutputs; i++){
					FieldName outputName = outputNames.get(i);

					updatedActiveFields.putAll(outputName, inputNames);

					Expression expression = complexTransformer.encode(i, inputNames);

					DerivedField derivedField = new DerivedField(opType, dataType)
						.setName(outputName)
						.setExpression(expression);

					localTransformations.addDerivedFields(derivedField);
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		if(dataFieldIt.hasNext()){
			Function<DataField, FieldName> function = new Function<DataField, FieldName>(){

				@Override
				public FieldName apply(DataField dataField){
					return dataField.getName();
				}
			};

			List<FieldName> unusedNames = Lists.newArrayList(Iterators.transform(dataFieldIt, function));

			logger.error("The list of mappings is shorter than the list of fields. Unused active fields: {}", unusedNames);

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

				ListIterator<MiningField> miningFieldIt = miningFields.listIterator();
				while(miningFieldIt.hasNext()){
					MiningField miningField = miningFieldIt.next();

					FieldName name = miningField.getName();

					if(updatedActiveFields.containsKey(name)){
						Set<FieldName> updatedNames = updatedActiveFields.get(name);

						Iterator<FieldName> updatedNameIt = updatedNames.iterator();

						miningField.setName(updatedNameIt.next());

						while(updatedNameIt.hasNext()){
							MiningField updatedMiningField = PMMLUtil.createMiningField(updatedNameIt.next());

							miningFieldIt.add(updatedMiningField);
						}
					}
				}

				Set<FieldName> names = new LinkedHashSet<>();

				miningFieldIt = miningFields.listIterator();
				while(miningFieldIt.hasNext()){
					MiningField miningField = miningFieldIt.next();

					FieldName name = miningField.getName();

					if(!names.add(name)){
						miningFieldIt.remove();
					}
				}

				return super.visit(miningSchema);
			}
		};

		activeFieldUpdater.applyTo(pmml);
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}

	static
	private FieldName getField(Object[] feature){
		return FieldName.create(getName(feature));
	}

	static
	private List<FieldName> getFieldList(Object[] feature, int expectedSize){
		List<String> names = getNameList(feature);

		if(names.size() != expectedSize){
			throw new IllegalArgumentException("Expected " + expectedSize + " element(s), got " + names.size() + " element(s)");
		}

		Function<String, FieldName> function = new Function<String, FieldName>(){

			@Override
			public FieldName apply(String name){
				return FieldName.create(name);
			}
		};

		return Lists.transform(names, function);
	}

	static
	private String getName(Object[] feature){

		try {
			if(feature[0] instanceof List){
				return (String)Iterables.getOnlyElement((List<?>)feature[0]);
			}

			return (String)feature[0];
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The key object (" + ClassDictUtil.formatClass(feature[0]) + ") is not a String", re);
		}
	}

	static
	private List<String> getNameList(Object[] feature){

		try {
			if(feature[0] instanceof List){
				return (List)feature[0];
			}

			return Collections.singletonList((String)feature[0]);
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The key object (" + ClassDictUtil.formatClass(feature[0]) + ") is not a String list", re);
		}
	}

	static
	private Transformer getTransformer(Object[] feature){

		try {
			return (Transformer)feature[1];
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The value object (" + ClassDictUtil.formatClass(feature[1]) + ") is not a Transformer or is not a supported Transformer subclass", re);
		}
	}

	static
	private void updateDataField(DataField dataField, FieldName name, Transformer transformer){
		dataField.setName(name)
			.setOpType(transformer.getOpType())
			.setDataType(transformer.getDataType());

		SchemaUtil.addValues(dataField, transformer.getClasses());
	}

	private static final Logger logger = LoggerFactory.getLogger(DataFrameMapper.class);
}