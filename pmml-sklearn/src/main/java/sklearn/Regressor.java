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

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Label;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.sklearn.SkLearnEncoder;

abstract
public class Regressor extends Estimator {

	public Regressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.REGRESSION;
	}

	@Override
	public boolean isSupervised(){
		return true;
	}

	@Override
	public int getNumberOfOutputs(){
		int numberOfOutputs = super.getNumberOfOutputs();

		if(numberOfOutputs == HasNumberOfOutputs.UNKNOWN){
			numberOfOutputs = 1;
		}

		return numberOfOutputs;
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){

		if(names.size() == 1){
			return encodeLabel(names.get(0), encoder);
		} else

		if(names.size() >= 2){
			List<Label> labels = new ArrayList<>();

			for(String name : names){
				Label label = encodeLabel(name, encoder);

				labels.add(label);
			}

			return new MultiLabel(labels);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	protected ScalarLabel encodeLabel(String name, SkLearnEncoder encoder){

		if(name != null){
			DataField dataField = encoder.createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);

			return new ContinuousLabel(dataField);
		} else

		{
			return new ContinuousLabel(DataType.DOUBLE);
		}
	}
}