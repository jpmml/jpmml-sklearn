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

import optbinning.BinnedFeature;
import optbinning.BinningProcess;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.scorecard.Attribute;
import org.dmg.pmml.scorecard.Characteristic;
import org.dmg.pmml.scorecard.Characteristics;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.BlockManager;
import pandas.core.DataFrame;
import pandas.core.Index;
import sklearn.Estimator;
import sklearn.HasClasses;
import sklearn.Regressor;

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
		String scalingMethod = getScalingMethod();

		SkLearnEncoder encoder = (SkLearnEncoder)schema.getEncoder();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		features = binningProcess.encode((List)features, encoder);

		schema = new Schema(encoder, label, features);

		if(scalingMethod != null){

			if(!(estimator instanceof Regressor)){
				throw new IllegalArgumentException();
			}

			return encodeScorecard(schema);
		}

		return estimator.encode(schema);
	}

	private org.dmg.pmml.scorecard.Scorecard encodeScorecard(Schema schema){
		DataFrame dfScorecard = getDFScorecard();
		Number intercept = getIntercept();

		BlockManager data = dfScorecard.getData();

		List<Index> axesArray = data.getAxesArray();
		if(axesArray.size() != 2){
			throw new IllegalArgumentException();
		}

		List<HasArray> blockValues = data.getBlockValues();

		// XXX
		HasArray pointsBlockValue = blockValues.get(blockValues.size() - 1);

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Characteristics characteristics = new Characteristics();

		List<?> pointValues = pointsBlockValue.getArrayContent();

		int index = 0;

		for(Feature feature : features){
			BinnedFeature binnedFeature = (BinnedFeature)feature;

			Characteristic characteristic = new Characteristic()
				.setName(binnedFeature.getName());

			List<Predicate> predicates = binnedFeature.getPredicates();
			for(Predicate predicate : predicates){

				if(predicate != null){
					Attribute attribute = new Attribute(predicate)
						.setPartialScore((Number)pointValues.get(index));

					characteristic.addAttributes(attribute);
				}

				index++;
			}

			characteristics.addCharacteristics(characteristic);
		}

		if(index != pointValues.size()){
			throw new IllegalArgumentException();
		}

		org.dmg.pmml.scorecard.Scorecard scorecard = new org.dmg.pmml.scorecard.Scorecard(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(label), characteristics)
			.setInitialScore(intercept)
			.setUseReasonCodes(false);

		return scorecard;
	}

	public BinningProcess getBinningProcess(){
		return get("binning_process_", BinningProcess.class);
	}

	public DataFrame getDFScorecard(){
		return get("_df_scorecard", DataFrame.class);
	}

	public Estimator getEstimator(){
		return get("estimator_", Estimator.class);
	}

	public Number getIntercept(){
		return getNumber("intercept_");
	}

	public String getScalingMethod(){
		return getOptionalString("scaling_method");
	}
}