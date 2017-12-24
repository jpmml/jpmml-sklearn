/*
 * Copyright (c) 2017 Villu Ruusmann
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
package tpot.builtins;

import java.util.List;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;

public class OutputUtil {

	private OutputUtil(){
	}

	static
	public OutputField getOutputField(Output output, FieldName name){

		if(output.hasOutputFields()){
			List<OutputField> outputFields = output.getOutputFields();

			for(OutputField outputField : outputFields){

				if((name).equals(outputField.getName())){
					return outputField;
				}
			}
		}

		return null;
	}
}