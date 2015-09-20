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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.CClassDict;
import sklearn.Transformer;
import sklearn.preprocessing.LabelEncoder;

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
		if(features.size() == 0 || features.size() != dataFields.size()){
			throw new IllegalArgumentException();
		}

		// Move the target column from the last position to the first position
		features.add(0, features.remove(features.size() - 1));

		final
		Map<FieldName, FieldName> transformedNames = new LinkedHashMap<>();

		for(int i = 0; i < dataFields.size(); i++){
			DataField dataField = dataFields.get(i);

			Object[] feature = features.get(i);

			if(feature[0] instanceof List){
				feature[0] = Iterables.getOnlyElement((List<?>)feature[0]);
			}

			FieldName name = FieldName.create((String)feature[0]);

			transformedNames.put(dataField.getName(), name);

			Transformer transformer = (Transformer)feature[1];

			if(i > 0){
				Expression expression = new FieldRef(name);

				if(transformer != null){
					expression = transformer.encode(name);
				}

				DerivedField derivedField = new DerivedField(dataField.getOpType(), dataField.getDataType())
					.setName(dataField.getName())
					.setExpression(expression);

				localTransformations.addDerivedFields(derivedField);

				// XXX
				if(transformer instanceof LabelEncoder){
					dataField.setOpType(OpType.CATEGORICAL)
						.setDataType(DataType.STRING);
				}
			}

			dataField.setName(name);
		}

		Visitor visitor = new AbstractVisitor(){

			@Override
			public VisitorAction visit(MiningField miningField){
				FieldName transformedName = transformedNames.get(miningField.getName());

				if(transformedName != null){
					miningField.setName(transformedName);
				}

				return VisitorAction.CONTINUE;
			}
		};
		visitor.applyTo(model);
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}
}