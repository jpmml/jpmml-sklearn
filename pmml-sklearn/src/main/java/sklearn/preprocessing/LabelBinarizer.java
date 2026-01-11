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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.HasSparseOutput;
import sklearn.SkLearnFields;
import sklearn.SkLearnTransformer;

public class LabelBinarizer extends SkLearnTransformer implements HasSparseOutput {

	public LabelBinarizer(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		List<?> classes = getClasses();

		return TypeUtil.getDataType(classes, DataType.STRING);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		List<?> classes = getClasses();

		Number negLabel = getNegLabel();
		Number posLabel = getPosLabel();

		Feature feature = SchemaUtil.getOnlyFeature(features);

		List<Object> categories = new ArrayList<>();
		categories.addAll(classes);

		List<Number> labelCategories = new ArrayList<>();
		labelCategories.add(negLabel);
		labelCategories.add(posLabel);

		List<Feature> result = new ArrayList<>();

		classes = prepareClasses(classes);

		for(int i = 0; i < classes.size(); i++){
			Object value = classes.get(i);

			if(ValueUtil.isZero(negLabel) && ValueUtil.isOne(posLabel)){
				result.add(new BinaryFeature(encoder, feature, value));
			} else

			{
				// "($name == value) ? pos_label : neg_label"
				Apply apply = ExpressionUtil.createApply(PMMLFunctions.IF,
					ExpressionUtil.createApply(PMMLFunctions.EQUAL, feature.ref(), ExpressionUtil.createConstant(feature.getDataType(), value)),
					ExpressionUtil.createConstant(posLabel),
					ExpressionUtil.createConstant(negLabel)
				);

				String name = (classes.size() > 1 ? createFieldName("labelBinarizer", feature, value) : createFieldName("labelBinarizer", feature));

				DerivedField derivedField = encoder.createDerivedField(name, apply);

				result.add(new CategoricalFeature(encoder, derivedField, labelCategories));
			}
		}

		encoder.toCategorical(feature.getName(), categories);

		return result;
	}

	protected List<?> prepareClasses(List<?> classes){

		// [negValue, posValue] -> [posValue]
		if(classes.size() == 2){
			return classes.subList(1, 2);
		} else

		if(classes.size() >= 3){
			return classes;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	public List<Object> getClasses(){
		return getObjectArray(SkLearnFields.CLASSES);
	}

	public Number getNegLabel(){
		return getNumber("neg_label");
	}

	public Number getPosLabel(){
		return getNumber("pos_label");
	}

	@Override
	public Boolean getSparseOutput(){
		return getBoolean("sparse_output");
	}
}