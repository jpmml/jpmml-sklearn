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

import java.util.List;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import sklearn.OneToManyTransformer;

public class LabelBinarizer extends OneToManyTransformer {

	public LabelBinarizer(String module, String name){
		super(module, name);
	}

	@Override
	public OpType getOpType(){
		return OpType.CATEGORICAL;
	}

	@Override
	public DataType getDataType(){
		return DataType.STRING;
	}

	@Override
	public int getNumberOfOutputs(){
		List<?> classes = getClasses();

		return classes.size();
	}

	@Override
	public Expression encode(int index, FieldName name){
		List<?> classes = getClasses();

		Object value = classes.get(index);

		Number posLabel = getPosLabel();
		Number negLabel = getNegLabel();

		if(Double.compare(posLabel.doubleValue(), 1d) == 0 && Double.compare(negLabel.doubleValue(), 0d) == 0){
			NormDiscrete normDiscrete = new NormDiscrete(name, String.valueOf(value));

			return normDiscrete;
		}

		// "($name == value) ? pos_label : neg_label"
		return PMMLUtil.createApply("if", PMMLUtil.createApply("equal", new FieldRef(name), PMMLUtil.createConstant(value)), PMMLUtil.createConstant(posLabel), PMMLUtil.createConstant(negLabel));
	}

	@Override
	public List<?> getClasses(){
		return (List)ClassDictUtil.getArray(this, "classes_");
	}

	public Number getPosLabel(){
		return (Number)get("pos_label");
	}

	public Number getNegLabel(){
		return (Number)get("neg_label");
	}
}