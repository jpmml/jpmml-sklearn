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
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class LabelBinarizer extends Transformer {

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

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

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
				Apply apply = PMMLUtil.createApply(PMMLFunctions.IF)
					.addExpressions(PMMLUtil.createApply(PMMLFunctions.EQUAL, feature.ref(), PMMLUtil.createConstant(value, feature.getDataType())))
					.addExpressions(PMMLUtil.createConstant(posLabel), PMMLUtil.createConstant(negLabel));

				FieldName name = (classes.size() > 1 ? createFieldName("labelBinarizer", feature, value) : createFieldName("label_binarizer", feature));

				DerivedField derivedField = encoder.createDerivedField(name, apply);

				result.add(new CategoricalFeature(encoder, derivedField, labelCategories));
			}
		}

		encoder.toCategorical(feature.getName(), categories);

		return result;
	}

	protected List<?> prepareClasses(List<?> classes){

		if(classes.size() < 2){
			throw new IllegalArgumentException();
		} else

		// [negValue, posValue] -> [posValue]
		if(classes.size() == 2){
			classes = classes.subList(1, 2);
		}

		return classes;
	}

	public List<?> getClasses(){
		return getArray("classes_");
	}

	public Number getPosLabel(){
		return getNumber("pos_label");
	}

	public Number getNegLabel(){
		return getNumber("neg_label");
	}
}