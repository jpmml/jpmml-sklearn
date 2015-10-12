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
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.CClassDict;
import sklearn.ComplexTransformer;
import sklearn.SimpleTransformer;
import sklearn.Transformer;

public class DataFrameMapper extends CClassDict {

	public DataFrameMapper(String module, String name){
		super(module, name);
	}

	public void updatePMML(PMML pmml){
		DataDictionary dataDictionary = pmml.getDataDictionary();

		List<DataField> dataFields = dataDictionary.getDataFields();

		Model model = Iterables.getOnlyElement(pmml.getModels());

		LocalTransformations localTransformations = model.getLocalTransformations();
		if(localTransformations == null){
			localTransformations = new LocalTransformations();

			model.setLocalTransformations(localTransformations);
		}

		List<Object[]> features = new ArrayList<>(getFeatures());
		if(features.size() < 1){
			throw new IllegalArgumentException();
		}

		// Move the target column from the last position to the first position
		features.add(0, features.remove(features.size() - 1));

		final
		Map<FieldName, FieldName> renamedFields = new LinkedHashMap<>();

		Iterator<DataField> it = dataFields.iterator();

		// The target column
		{
			Object[] feature = features.get(0);

			Transformer transformer = getTransformer(feature);
			if(transformer != null){
				throw new IllegalArgumentException();
			}

			FieldName name = FieldName.create(getName(feature));

			DataField dataField = it.next();

			renamedFields.put(dataField.getName(), name);
		}

		// Zero or more active columns
		for(int i = 1; i < features.size(); i++){
			Object[] feature = features.get(i);

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

				FieldName name = FieldName.create(getName(feature));

				DataField dataField = it.next();

				renamedFields.put(dataField.getName(), name);

				Expression expression = simpleTransformer.encode(name);

				DerivedField derivedField = encodeDerivedField(dataField, expression);

				localTransformations.addDerivedFields(derivedField);

				dataField.setOpType(simpleTransformer.getOpType())
					.setDataType(simpleTransformer.getDataType());
			} else

			if(transformer instanceof ComplexTransformer){
				ComplexTransformer complexTransformer = (ComplexTransformer)transformer;

				for(int j = 0; j < complexTransformer.getNumberOfFeatures(); j++){
					throw new UnsupportedOperationException();
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		if(it.hasNext()){
			throw new IllegalArgumentException();
		}

		Visitor visitor = new AbstractVisitor(){

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
		visitor.applyTo(pmml);
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}

	static
	private String getName(Object[] feature){

		if(feature[0] instanceof List){
			return (String)Iterables.getOnlyElement((List<?>)feature[0]);
		}

		return (String)feature[0];
	}

	static
	private Transformer getTransformer(Object[] feature){
		return (Transformer)feature[1];
	}

	static
	private DerivedField encodeDerivedField(DataField dataField, Expression expression){
		DerivedField derivedField = new DerivedField(dataField.getOpType(), dataField.getDataType())
			.setName(dataField.getName())
			.setExpression(expression);

		return derivedField;
	}
}