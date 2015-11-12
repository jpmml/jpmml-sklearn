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
package sklearn;

import java.util.Collections;
import java.util.List;

import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;

abstract
public class Transformer extends ClassDict {

	public Transformer(String module, String name){
		super(module, name);
	}

	abstract
	public int getNumberOfInputs();

	abstract
	public int getNumberOfOutputs();

	abstract
	public Expression encode(int index, List<FieldName> names);

	public OpType getOpType(){
		return OpType.CONTINUOUS;
	}

	public DataType getDataType(){
		return DataType.DOUBLE;
	}

	public List<?> getClasses(){
		return Collections.emptyList();
	}
}