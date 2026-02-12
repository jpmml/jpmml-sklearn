/*
 * Copyright (c) 2026 Villu Ruusmann
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
package ngboost;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Interval;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import sklearn.EstimatorCastFunction;
import sklearn.Regressor;

public class NGBRegressor extends Regressor {

	public NGBRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		String distName = getDistName();

		switch(distName){
			case NGBRegressor.DIST_NORMAL:
				return encodeNormalModel(schema);
			case NGBRegressor.DIST_POISSON:
				return encodePoissonModel(schema);
			default:
				throw new IllegalArgumentException(distName);
		}
	}

	public MiningModel encodeNormalModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		ClassDictUtil.checkSize(baseModels, scalings);

		if(!isWeighted(scalings)){
			scalings = null;
		}

		PMMLEncoder encoder = schema.getEncoder();
		ContinuousLabel continuousLabel = schema.requireContinuousLabel();

		DataField ciDataField = encoder.createDataField(NGBoostNames.INPUT_CI, OpType.CONTINUOUS, DataType.DOUBLE)
			.setDisplayName("Confidence level")
			.addIntervals(new Interval(Interval.Closure.OPEN_OPEN, 0, 1));

		Feature ciFeature = new ContinuousFeature(encoder, ciDataField);

		DefineFunction defineFunction = encodeZCriticalFunction();

		encoder.addDefineFunction(defineFunction);

		Schema segmentSchema = schema.toAnonymousSchema();

		ContinuousLabel segmentLabel = segmentSchema.requireContinuousLabel();

		MiningModel locModel = encodeParamModel(0, baseModels, scalings, segmentSchema)
			.setTargets(ModelUtil.createRescaleTargets(-learningRate.doubleValue(), initParams.get(0), segmentLabel));

		Feature locFeature = encodePredictedParam(NGBoostNames.OUTPUT_LOC, locModel, encoder);

		MiningModel scaleModel = encodeParamModel(1, baseModels, scalings, segmentSchema)
			.setTargets(ModelUtil.createRescaleTargets(-learningRate.doubleValue(), initParams.get(1), segmentLabel));

		Feature scaleFeature = encodePredictedParam(NGBoostNames.OUTPUT_SCALE, scaleModel, encoder);

		RegressionModel simpleRegressionModel = encodeRegression(locFeature, RegressionModel.NormalizationMethod.NONE, schema);

		// XXX: Shared between two expressions
		Apply boundApply = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY,
			ExpressionUtil.createApply(PMMLFunctions.EXP, scaleFeature.ref()),
			ExpressionUtil.createApply(defineFunction, ciFeature.ref())
		);

		OutputField lowerBoundOutputField = new OutputField(NGBoostNames.createLowerBound(continuousLabel.getName()), OpType.CONTINUOUS, DataType.DOUBLE)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, locFeature.ref(), boundApply));

		OutputField upperBoundOutputField = new OutputField(NGBoostNames.createUpperBound(continuousLabel.getName()), OpType.CONTINUOUS, DataType.DOUBLE)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(ExpressionUtil.createApply(PMMLFunctions.ADD, locFeature.ref(), boundApply));

		Output output = new Output()
			.addOutputFields(lowerBoundOutputField, upperBoundOutputField);

		RegressionModel complexRegressionModel = encodeRegression(locFeature, RegressionModel.NormalizationMethod.NONE, schema)
			.setOutput(output);

		Segmentation segmentation = new Segmentation(Segmentation.MultipleModelMethod.MODEL_CHAIN, null)
			.addSegments(
				new Segment(True.INSTANCE, locModel),
				new Segment(new SimplePredicate(ciDataField, SimplePredicate.Operator.IS_NOT_MISSING, null), scaleModel),
				new Segment(new SimplePredicate(ciDataField, SimplePredicate.Operator.IS_MISSING, null), simpleRegressionModel),
				new Segment(new SimplePredicate(ciDataField, SimplePredicate.Operator.IS_NOT_MISSING, null), complexRegressionModel)
			);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(segmentation);

		return miningModel;
	}

	public MiningModel encodePoissonModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		ClassDictUtil.checkSize(baseModels, scalings);

		PMMLEncoder encoder = schema.getEncoder();
		Schema segmentSchema = schema.toAnonymousSchema();

		ContinuousLabel segmentLabel = segmentSchema.requireContinuousLabel();

		MiningModel locModel = encodeParamModel(0, baseModels, scalings, segmentSchema)
			.setTargets(ModelUtil.createRescaleTargets(-learningRate.doubleValue(), initParams.get(0), segmentLabel));

		Feature locFeature = encodePredictedParam(NGBoostNames.OUTPUT_LOC, locModel, encoder);

		RegressionModel regressionModel = encodeRegression(locFeature, RegressionModel.NormalizationMethod.EXP, schema);

		return MiningModelUtil.createModelChain(Arrays.asList(locModel, regressionModel), Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	public List<List<Regressor>> getBaseModels(){
		CastFunction<List> castFunction = new CastFunction<List>(List.class){

			@Override
			public List<Regressor> apply(Object object){
				List<?> values = super.apply(object);

				CastFunction<Regressor> castFunction = new EstimatorCastFunction<>(Regressor.class);

				return Lists.transform(values, castFunction);
			}
		};

		return (List)getList("base_models", castFunction);
	}

	public String getDistName(){
		return getEnum("Dist_name", this::getString, Arrays.asList(NGBRegressor.DIST_NORMAL, NGBRegressor.DIST_POISSON));
	}

	public List<Number> getInitParams(){
		return getNumberListLike("init_params");
	}

	public Number getLearningRate(){
		return getNumber("learning_rate");
	}

	public List<Number> getScalings(){
		return getNumberListLike("scalings");
	}

	static
	public boolean isWeighted(List<Number> scalings){

		for(Number scaling : scalings){

			if(scaling.doubleValue() != 1d){
				return true;
			}
		}

		return false;
	}

	static
	private MiningModel encodeParamModel(int index, List<List<Regressor>> baseModels, List<Number> scalings, Schema schema){
		List<Model> models = new ArrayList<>();

		for(int i = 0; i < baseModels.size(); i++){
			Regressor regressor = (baseModels.get(i)).get(index);

			Model model = regressor.encodeModel(schema);

			models.add(model);
		}

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(MiningModelUtil.createSegmentation((scalings != null ? Segmentation.MultipleModelMethod.WEIGHTED_SUM : Segmentation.MultipleModelMethod.SUM), Segmentation.MissingPredictionTreatment.RETURN_MISSING, models, scalings));

		return miningModel;
	}

	static
	private ContinuousFeature encodePredictedParam(String name, Model model, PMMLEncoder encoder){
		OutputField outputField = ModelUtil.createPredictedField(name, OpType.CONTINUOUS, DataType.DOUBLE);

		DerivedOutputField derivedField = encoder.createDerivedField(model, outputField, true);

		return new ContinuousFeature(encoder, derivedField);
	}

	static
	private RegressionModel encodeRegression(Feature locFeature, RegressionModel.NormalizationMethod normalizationMethod, Schema schema){
		RegressionModel regressionModel = RegressionModelUtil.createRegression(Collections.singletonList(locFeature), Collections.singletonList(1d), null, normalizationMethod, schema);

		return regressionModel;
	}

	static
	private DefineFunction encodeZCriticalFunction(){
		ParameterField valueField = new ParameterField("p")
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.STDNORMALIDF,
			ExpressionUtil.createApply(PMMLFunctions.ADD,
				ExpressionUtil.createConstant(0.5d),
				ExpressionUtil.createApply(PMMLFunctions.DIVIDE, new FieldRef(valueField), ExpressionUtil.createConstant(2d))
			)
		);

		DefineFunction defineFunction = new DefineFunction("zCritical", OpType.CONTINUOUS, DataType.DOUBLE, null, apply)
			.addParameterFields(valueField);

		return defineFunction;
	}

	private static final String DIST_NORMAL = "Normal";
	private static final String DIST_POISSON = "Poisson";
}