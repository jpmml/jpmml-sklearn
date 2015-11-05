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
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.BayesOutput;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.ClassDictUtil;
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

	public void updatePMML(PMML pmml){
		List<Object[]> features = new ArrayList<>(getFeatures());

		if(features.size() < 1){
			logger.warn("The list of mappings is empty");

			return;
		}

		// Move the target column from the last position to the first position
		features.add(0, features.remove(features.size() - 1));

		DataDictionary dataDictionary = pmml.getDataDictionary();

		List<DataField> dataFields = dataDictionary.getDataFields();

		logger.info("Re-mapping {} target field and {} active field(s)", 1, (dataFields.size() - 1));

		Model model = Iterables.getOnlyElement(pmml.getModels());

		LocalTransformations localTransformations = model.getLocalTransformations();
		if(localTransformations == null){
			localTransformations = new LocalTransformations();

			model.setLocalTransformations(localTransformations);
		}

		final
		Map<FieldName, FieldName> renamedFields = new LinkedHashMap<>();

		Iterator<DataField> it = dataFields.iterator();

		// The target column
		{
			Object[] feature = features.get(0);

			FieldName name = FieldName.create(getName(feature));

			Transformer transformer = getTransformer(feature);
			if(transformer != null){
				logger.error("Target field {} must not specify a transformation", name);

				throw new IllegalArgumentException();
			}

			DataField dataField = it.next();

			logger.info("Renaming target field {} to {}", dataField.getName(), name);

			renamedFields.put(dataField.getName(), name);
		}

		// Zero or more active columns
		for(int i = 1; i < features.size(); i++){
			Object[] feature = features.get(i);

			FieldName name = FieldName.create(getName(feature));

			Transformer transformer = getTransformer(feature);

			if(transformer == null){
				transformer = new SimpleTransformer(null, null){

					@Override
					public Expression encode(FieldName name){
						FieldRef fieldRef = new FieldRef(name);

						return fieldRef;
					}
				};
			}

			DataField dataField = it.next();

			logger.info("Renaming active field {} to {}", dataField.getName(), name);

			renamedFields.put(dataField.getName(), name);

			if(transformer instanceof SimpleTransformer){
				SimpleTransformer simpleTransformer = (SimpleTransformer)transformer;

				Expression expression = simpleTransformer.encode(name);

				DerivedField derivedField = encodeDerivedField(dataField, expression);

				localTransformations.addDerivedFields(derivedField);
			} else

			if(transformer instanceof ComplexTransformer){
				ComplexTransformer complexTransformer = (ComplexTransformer)transformer;

				for(int j = 0; j < complexTransformer.getNumberOfFeatures(); j++){
					Expression expression = complexTransformer.encode(j, name);

					DataField elementDataField = dataField;

					if(j > 0){
						elementDataField = it.next();

						logger.info("Renaming active field {} to {}", elementDataField.getName(), name);

						renamedFields.put(elementDataField.getName(), name);

						it.remove();
					}

					DerivedField derivedField = encodeDerivedField(elementDataField, expression);

					localTransformations.addDerivedFields(derivedField);
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			dataField.setOpType(transformer.getOpType())
				.setDataType(transformer.getDataType());

			SchemaUtil.addValues(dataField, transformer.getClasses());
		}

		if(it.hasNext()){
			Function<DataField, FieldName> function = new Function<DataField, FieldName>(){

				@Override
				public FieldName apply(DataField dataField){
					return dataField.getName();
				}
			};

			List<FieldName> unusedNames = Lists.newArrayList(Iterators.transform(it, function));

			logger.error("The list of mappings is shorter than the list of fields. Unused active fields: {}", unusedNames);

			throw new IllegalArgumentException();
		}

		Visitor fieldRenamer = new AbstractVisitor(){

			@Override
			public VisitorAction visit(BayesOutput bayesOutput){
				bayesOutput.setFieldName(filterName(bayesOutput.getFieldName()));

				return super.visit(bayesOutput);
			}

			@Override
			public VisitorAction visit(DataField dataField){
				dataField.setName(filterName(dataField.getName()));

				return super.visit(dataField);
			}

			@Override
			public VisitorAction visit(MiningField miningField){
				miningField.setName(filterName(miningField.getName()));

				return super.visit(miningField);
			}

			private FieldName filterName(FieldName name){

				if(renamedFields.containsKey(name)){
					return renamedFields.get(name);
				}

				return name;
			}
		};

		fieldRenamer.applyTo(pmml);

		Visitor miningSchemaPruner = new AbstractVisitor(){

			@Override
			public VisitorAction visit(MiningSchema miningSchema){
				Set<FieldName> names = new LinkedHashSet<>();

				List<MiningField> miningFields = miningSchema.getMiningFields();

				for(Iterator<MiningField> it = miningFields.iterator(); it.hasNext(); ){
					MiningField miningField = it.next();

					FieldName name = miningField.getName();

					if(!names.add(name)){
						it.remove();
					}
				}

				return super.visit(miningSchema);
			}
		};

		miningSchemaPruner.applyTo(pmml);
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
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
	private Transformer getTransformer(Object[] feature){

		try {
			return (Transformer)feature[1];
		} catch(RuntimeException re){
			throw new IllegalArgumentException("The value object (" + ClassDictUtil.formatClass(feature[1]) + ") is not a Transformer or is not a supported Transformer subclass", re);
		}
	}

	static
	private DerivedField encodeDerivedField(DataField dataField, Expression expression){
		DerivedField derivedField = new DerivedField(dataField.getOpType(), dataField.getDataType())
			.setName(dataField.getName())
			.setExpression(expression);

		return derivedField;
	}

	private static final Logger logger = LoggerFactory.getLogger(DataFrameMapper.class);
}