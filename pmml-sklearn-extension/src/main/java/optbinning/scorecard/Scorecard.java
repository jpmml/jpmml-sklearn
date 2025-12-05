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
import java.util.stream.Collectors;

import numpy.core.TypeDescriptor;
import optbinning.BinnedFeature;
import optbinning.BinningProcess;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.scorecard.Attribute;
import org.dmg.pmml.scorecard.Characteristic;
import org.dmg.pmml.scorecard.Characteristics;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import pandas.core.BlockManager;
import pandas.core.DataFrame;
import pandas.core.Index;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.HasClasses;

public class Scorecard extends Estimator implements HasClasses {

	public Scorecard(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		Estimator estimator = getEstimator();
		String scalingMethod = getScalingMethod();

		if(scalingMethod != null){
			return MiningFunction.REGRESSION;
		}

		return estimator.getMiningFunction();
	}

	@Override
	public boolean isSupervised(){
		return true;
	}

	@Override
	public int getNumberOfOutputs(){
		return 1;
	}

	@Override
	public List<?> getClasses(){
		Estimator estimator = getEstimator();

		return EstimatorUtil.getClasses(estimator);
	}

	@Override
	public boolean hasProbabilityDistribution(){
		Estimator estimator = getEstimator();

		return EstimatorUtil.hasProbabilityDistribution(estimator);
	}

	@Override
	public Label encodeLabel(List<String> names, SkLearnEncoder encoder){
		Estimator estimator = getEstimator();
		String scalingMethod = getScalingMethod();

		ClassDictUtil.checkSize(1, names);

		if(scalingMethod != null){
			String name = names.get(0);

			if(name != null){
				DataField dataField = encoder.createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);

				return new ContinuousLabel(dataField);
			} else

			{
				return new ContinuousLabel(DataType.DOUBLE);
			}
		}

		return estimator.encodeLabel(names, encoder);
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
			return encodeScorecard(schema);
		}

		return estimator.encode(schema);
	}

	private org.dmg.pmml.scorecard.Scorecard encodeScorecard(Schema schema){
		DataFrame dfScorecard = getDFScorecard();
		Number intercept = getIntercept();

		BlockManager data = dfScorecard.getData();

		Index rowAxis = data.getRowAxis();
		List<?> rowValues = rowAxis.getArrayContent();

		List<HasArray> blockValues = data.getBlockValues();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Characteristics characteristics = new Characteristics();

		HasArray pointBlockValues = getPoints(blockValues);

		List<?> pointValues = pointBlockValues.getArrayContent();
		if(pointValues.size() > rowValues.size()){
			pointValues = pointValues.subList(pointValues.size() - rowValues.size(), pointValues.size());
		}

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
		return getEstimator("estimator_");
	}

	public Number getIntercept(){
		return getNumber("intercept_");
	}

	public String getScalingMethod(){
		return getOptionalString("scaling_method");
	}

	static
	private HasArray getPoints(List<HasArray> blockValues){
		blockValues = blockValues.stream()
			.filter(blockValue -> {
				Object descr = blockValue.getArrayType();

				TypeDescriptor typeDescriptor = new TypeDescriptor(descr);

				DataType dataType = typeDescriptor.getDataType();

				return (dataType == DataType.DOUBLE);
			})
			.collect(Collectors.toList());

		return blockValues.get(blockValues.size() - 1);
	}
}