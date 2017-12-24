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

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.PMMLEncoder;

public class ContinuousOutputFeature extends ContinuousFeature {

	private Output output = null;


	public ContinuousOutputFeature(PMMLEncoder encocder, Output output, OutputField outputField){
		this(encocder, output, outputField.getName(), outputField.getDataType());
	}

	public ContinuousOutputFeature(PMMLEncoder encoder, Output output, FieldName name, DataType dataType){
		super(encoder, name, dataType);

		setOutput(output);
	}

	@Override
	public ContinuousOutputFeature toContinuousFeature(){
		return this;
	}

	@Override
	public ContinuousOutputFeature toContinuousFeature(DataType dataType){
		ContinuousOutputFeature continuousFeature = toContinuousFeature();

		if((dataType).equals(continuousFeature.getDataType())){
			return continuousFeature;
		}

		PMMLEncoder encoder = ensureEncoder();

		FieldName name = FieldName.create((dataType.name()).toLowerCase() + "(" + (continuousFeature.getName()).getValue() + ")");

		Output output = getOutput();

		OutputField outputField = OutputUtil.getOutputField(output, name);
		if(outputField == null){
			outputField = new OutputField(name, dataType)
				.setOpType(OpType.CONTINUOUS)
				.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
				.setFinalResult(false)
				.setExpression(continuousFeature.ref());

			output.addOutputFields(outputField);
		}

		return new ContinuousOutputFeature(encoder, output, outputField.getName(), outputField.getDataType());
	}

	public Output getOutput(){
		return this.output;
	}

	private void setOutput(Output output){

		if(output == null){
			throw new IllegalArgumentException();
		}

		this.output = output;
	}
}