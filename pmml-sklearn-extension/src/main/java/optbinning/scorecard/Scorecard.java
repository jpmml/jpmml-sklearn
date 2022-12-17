/*
 * Copyright (c) 2022 Villu Ruusmann
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
package optbinning.scorecard;

import java.util.List;

import optbinning.BinningProcess;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.Schema;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Estimator;
import sklearn.HasClasses;

public class Scorecard extends Estimator implements HasClasses {

	public Scorecard(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		Estimator estimator = getEstimator();

		return estimator.getMiningFunction();
	}

	@Override
	public List<?> getClasses(){
		Estimator estimator = getEstimator();

		if(estimator instanceof HasClasses){
			HasClasses hasClasses = (HasClasses)estimator;

			return hasClasses.getClasses();
		}

		throw new UnsupportedOperationException();
	}

	@Override
	public Model encodeModel(Schema schema){
		BinningProcess binningProcess = getBinningProcess();
		Estimator estimator = getEstimator();

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		features = binningProcess.encode((List)features, encoder);

		schema = new Schema(encoder, label, features);

		return estimator.encode(schema);
	}

	public BinningProcess getBinningProcess(){
		return get("binning_process_", BinningProcess.class);
	}

	public Estimator getEstimator(){
		return get("estimator_", Estimator.class);
	}
}