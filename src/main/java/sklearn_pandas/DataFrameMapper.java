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

import java.lang.reflect.Field;
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
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.ReflectionUtil;
import org.jpmml.model.visitors.AbstractSimpleVisitor;
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
		Map<FieldName, FieldName> renamedFields = new LinkedHashMap<>();

		final
		Map<FieldName, FieldName> transformedFields = new LinkedHashMap<>();

		for(int i = 0; i < dataFields.size(); i++){
			DataField dataField = dataFields.get(i);

			Object[] feature = features.get(i);

			if(feature[0] instanceof List){
				feature[0] = Iterables.getOnlyElement((List<?>)feature[0]);
			}

			FieldName name = FieldName.create((String)feature[0]);

			renamedFields.put(dataField.getName(), name);

			if(i > 0){
				Transformer transformer = (Transformer)feature[1];

				if(transformer != null){
					FieldName derivedName = FieldName.create("derived_" + name.getValue());

					Expression expression = transformer.encode(name);

					DerivedField derivedField = new DerivedField(dataField.getOpType(), dataField.getDataType())
						.setName(derivedName)
						.setExpression(expression);

					localTransformations.addDerivedFields(derivedField);

					// XXX
					if(transformer instanceof LabelEncoder){
						dataField.setOpType(OpType.CATEGORICAL)
							.setDataType(DataType.STRING);
					}

					transformedFields.put(dataField.getName(), derivedName);
				}
			}
		}

		Visitor visitor = new AbstractSimpleVisitor(){

			@Override
			public VisitorAction visit(PMMLObject object){
				List<Field> fields = ReflectionUtil.getAllInstanceFields(object);

				for(Field field : fields){
					Object value = ReflectionUtil.getFieldValue(field, object);

					if(value instanceof FieldName){
						FieldName name = (FieldName)value;

						FieldName updatedName = name;

						if(renamedFields.containsKey(name)){
							updatedName = renamedFields.get(name);
						} // End if

						if(!(object instanceof DataField || object instanceof MiningField)){

							if(transformedFields.containsKey(name)){
								updatedName = transformedFields.get(name);
							}
						}

						ReflectionUtil.setFieldValue(field, object, updatedName);
					}
				}

				return VisitorAction.CONTINUE;
			}
		};
		visitor.applyTo(pmml);
	}

	public List<Object[]> getFeatures(){
		return (List)get("features");
	}
}