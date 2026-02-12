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

import com.google.common.collect.Iterables;
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
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.converter.transformations.ExpTransformation;
import org.jpmml.model.InvalidElementException;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import sklearn.EstimatorCastFunction;
import sklearn.Regressor;

public class NGBRegressor extends Regressor {

	public NGBRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		@SuppressWarnings("unused")
		String distName = getDistName();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		ClassDictUtil.checkSize(baseModels, scalings);

		if(!isWeighted(scalings)){
			scalings = null;
		}

		PMMLEncoder encoder = schema.getEncoder();
		ContinuousLabel continuousLabel = schema.requireContinuousLabel();

		DataField ciDataField = encoder.createDataField("ci", OpType.CONTINUOUS, DataType.DOUBLE)
			.setDisplayName("Confidence level")
			.addIntervals(new Interval(Interval.Closure.OPEN_OPEN, 0, 1));

		Schema segmentSchema = schema.toAnonymousSchema();

		ContinuousLabel regressionLabel = new ContinuousLabel(null, DataType.DOUBLE);

		MiningModel locModel = encodeParamModel(0, baseModels, scalings, segmentSchema)
			.setTargets(ModelUtil.createRescaleTargets(-learningRate.doubleValue(), initParams.get(0), regressionLabel))
			.setOutput(ModelUtil.createPredictedOutput("loc", OpType.CONTINUOUS, DataType.DOUBLE));

		Segment locSegment = new Segment(True.INSTANCE, locModel)
			.setId("loc");

		Feature muFeature = new ContinuousFeature(encoder, getFinalOutputField(locModel));

		MiningModel scaleModel = encodeParamModel(1, baseModels, scalings, segmentSchema)
			.setTargets(ModelUtil.createRescaleTargets(-learningRate.doubleValue(), initParams.get(1), regressionLabel))
			.setOutput(ModelUtil.createPredictedOutput("scale", OpType.CONTINUOUS, DataType.DOUBLE, new ExpTransformation()));

		Segment scaleSegment = new Segment(new SimplePredicate(ciDataField, SimplePredicate.Operator.IS_NOT_MISSING, null), scaleModel)
			.setId("scale");

		Feature sigmaFeature = new ContinuousFeature(encoder, getFinalOutputField(scaleModel));

		DefineFunction zScoreFunction = encodeZScoreFunction();

		encoder.addDefineFunction(zScoreFunction);

		OutputField lowerBoundOutputField = new OutputField(FieldNameUtil.create("lower", continuousLabel.getName()), OpType.CONTINUOUS, DataType.DOUBLE)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(encodeTargetBound(new FieldRef(ciDataField), muFeature.ref(), sigmaFeature.ref(), -2d));

		OutputField upperBoundOutputField = new OutputField(FieldNameUtil.create("upper", continuousLabel.getName()), OpType.CONTINUOUS, DataType.DOUBLE)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(encodeTargetBound(new FieldRef(ciDataField), muFeature.ref(), sigmaFeature.ref(), 2d));

		Output output = new Output()
			.addOutputFields(lowerBoundOutputField, upperBoundOutputField);

		RegressionModel ngboostModel = RegressionModelUtil.createRegression(Collections.singletonList(muFeature), Collections.singletonList(1d), null, RegressionModel.NormalizationMethod.NONE, schema)
			.setOutput(output);

		Segment ngboostSegment = new Segment(True.INSTANCE, ngboostModel)
			.setId("ngboost");

		Segmentation segmentation = new Segmentation(Segmentation.MultipleModelMethod.MODEL_CHAIN, null)
			.addSegments(locSegment, scaleSegment, ngboostSegment);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(segmentation);

		return miningModel;
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
		return getEnum("Dist_name", this::getString, Arrays.asList(NGBRegressor.DIST_NORMAL));
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
	private DefineFunction encodeZScoreFunction(){
		ParameterField ciField = new ParameterField("ci")
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		ParameterField divisorField = new ParameterField("divisor")
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.STDNORMALIDF,
			ExpressionUtil.createApply(PMMLFunctions.ADD,
				ExpressionUtil.createConstant(0.5d),
				ExpressionUtil.createApply(PMMLFunctions.DIVIDE, new FieldRef(ciField), new FieldRef(divisorField))
			)
		);

		DefineFunction defineFunction = new DefineFunction("zScore", OpType.CONTINUOUS, DataType.DOUBLE, null, apply)
			.addParameterFields(ciField, divisorField);

		return defineFunction;
	}

	static
	private Apply encodeTargetBound(FieldRef ciFieldRef, FieldRef muFieldRef, FieldRef sigmaFieldRef, double divisor){
		Apply apply = ExpressionUtil.createApply(PMMLFunctions.IF,
			ExpressionUtil.createApply(PMMLFunctions.ISNOTMISSING, ciFieldRef),
			ExpressionUtil.createApply(PMMLFunctions.ADD,
				muFieldRef,
				ExpressionUtil.createApply(PMMLFunctions.MULTIPLY,
					ExpressionUtil.createApply("zScore", ciFieldRef, ExpressionUtil.createConstant(divisor)),
					sigmaFieldRef
				)
			)
		);

		return apply;
	}

	static
	private OutputField getFinalOutputField(Model model){
		Output output = model.getOutput();

		if(output == null || !output.hasOutputFields()){
			throw new InvalidElementException(model);
		}

		List<OutputField> outputFields = output.getOutputFields();

		return Iterables.getLast(outputFields);
	}

	private static final String DIST_NORMAL = "Normal";
}