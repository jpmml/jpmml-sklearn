/*
 * Copyright (c) 2019 Villu Ruusmann
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
package sklearn.pipeline;

import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Schema;
import sklearn.Regressor;

public class PipelineRegressor extends Regressor {

	private Pipeline pipeline = null;


	public PipelineRegressor(Pipeline pipeline){
		super(pipeline.getPythonModule(), pipeline.getPythonName());

		setPipeline(pipeline);
	}

	@Override
	public int getNumberOfFeatures(){
		Pipeline pipeline = getPipeline();

		return pipeline.getNumberOfFeatures();
	}

	@Override
	public OpType getOpType(){
		Pipeline pipeline = getPipeline();

		return pipeline.getOpType();
	}

	@Override
	public DataType getDataType(){
		Pipeline pipeline = getPipeline();

		return pipeline.getDataType();
	}

	@Override
	public boolean isSupervised(){
		Regressor regressor = getFinalRegressor();

		return regressor.isSupervised();
	}

	@Override
	public Model encodeModel(Schema schema){
		Pipeline pipeline = getPipeline();

		return pipeline.encodeModel(schema);
	}

	public Regressor getFinalRegressor(){
		Pipeline pipeline = getPipeline();

		return pipeline.getFinalEstimator(Regressor.class);
	}

	public Pipeline getPipeline(){
		return this.pipeline;
	}

	private void setPipeline(Pipeline pipeline){
		this.pipeline = pipeline;
	}
}