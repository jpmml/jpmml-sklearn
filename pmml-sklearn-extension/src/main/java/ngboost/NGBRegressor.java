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
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import net.razorvine.pickle.objects.ClassDictConstructor;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Interval;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.python.ClassDictConstructorUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnException;
import sklearn.Regressor;

public class NGBRegressor extends Regressor implements HasNGBoostOptions {

	public NGBRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		String distName = getDistName();

		switch(distName){
			case NGBRegressor.DIST_LOGNORMAL:
				return encodeLogNormalModel(schema);
			case NGBRegressor.DIST_NORMAL:
				return encodeNormalModel(schema);
			case NGBRegressor.DIST_POISSON:
				return encodePoissonModel(schema);
			default:
				throw new IllegalArgumentException(distName);
		}
	}

	/**
	 * @see NGBoostNames#OUTPUT_LOC
	 * @see NGBoostNames#OUTPUT_SCALE
	 */
	public MiningModel encodeLogNormalModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		PMMLEncoder encoder = schema.getEncoder();

		Object confidenceLevel = getOption(HasNGBoostOptions.OPTION_CONFIDENCE_LEVEL, false);

		Expression confidenceLevelExpression = encodeConfidenceLevel(confidenceLevel);

		Schema segmentSchema = schema.toAnonymousSchema();

		MiningModel locModel = NGBoostUtil.encodeParamModel(0, initParams, baseModels, scalings, learningRate, segmentSchema);

		ContinuousFeature locFeature = NGBoostUtil.encodePredictedParam(NGBoostNames.OUTPUT_LOC, locModel, encoder);

		MiningModel scaleModel = NGBoostUtil.encodeParamModel(1, initParams, baseModels, scalings, learningRate, segmentSchema);

		ContinuousFeature scaleFeature = NGBoostUtil.encodePredictedParam(NGBoostNames.OUTPUT_SCALE, scaleModel, encoder);

		Output boundsOutput = null;

		if(confidenceLevelExpression != null){
			boundsOutput = encodeBoundsOutput(locFeature, scaleFeature, confidenceLevelExpression, PMMLFunctions.EXP, schema);
		}

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.ADD,
			locFeature.ref(),
			ExpressionUtil.createApply(PMMLFunctions.DIVIDE,
				ExpressionUtil.createApply(PMMLFunctions.EXP,
					ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, ExpressionUtil.createConstant(2d), scaleFeature.ref())
				),
				ExpressionUtil.createConstant(2d)
			)
		);

		DerivedField derivedField = encoder.createDerivedField("logMean", OpType.CONTINUOUS, DataType.DOUBLE, apply);

		ContinuousFeature logMeanFeature = new ContinuousFeature(encoder, derivedField);

		RegressionModel regressionModel = NGBoostUtil.encodeRegression(logMeanFeature, RegressionModel.NormalizationMethod.EXP, schema)
			.setOutput(boundsOutput);

		return MiningModelUtil.createModelChain(Arrays.asList(locModel, scaleModel, regressionModel), Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	/**
	 * @see NGBoostNames#OUTPUT_LOC
	 * @see NGBoostNames#OUTPUT_SCALE
	 */
	public MiningModel encodeNormalModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		PMMLEncoder encoder = schema.getEncoder();

		Object confidenceLevel = getOption(HasNGBoostOptions.OPTION_CONFIDENCE_LEVEL, false);

		Expression confidenceLevelExpression = encodeConfidenceLevel(confidenceLevel);

		Schema segmentSchema = schema.toAnonymousSchema();

		List<Model> models = new ArrayList<>();

		MiningModel locModel = NGBoostUtil.encodeParamModel(0, initParams, baseModels, scalings, learningRate, segmentSchema);

		ContinuousFeature locFeature = NGBoostUtil.encodePredictedParam(NGBoostNames.OUTPUT_LOC, locModel, encoder);

		models.add(locModel);

		Output boundsOutput = null;

		if(confidenceLevelExpression != null){
			MiningModel scaleModel = NGBoostUtil.encodeParamModel(1, initParams, baseModels, scalings, learningRate, segmentSchema);

			ContinuousFeature scaleFeature = NGBoostUtil.encodePredictedParam(NGBoostNames.OUTPUT_SCALE, scaleModel, encoder);

			models.add(scaleModel);

			boundsOutput = encodeBoundsOutput(locFeature, scaleFeature, confidenceLevelExpression, null, schema);
		}

		RegressionModel regressionModel = NGBoostUtil.encodeRegression(locFeature, RegressionModel.NormalizationMethod.NONE, schema)
			.setOutput(boundsOutput);

		models.add(regressionModel);

		return MiningModelUtil.createModelChain(models, Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	/**
	 * @see NGBoostNames#OUTPUT_LOC
	 */
	public MiningModel encodePoissonModel(Schema schema){
		List<List<Regressor>> baseModels = getBaseModels();
		List<Number> initParams = getInitParams();
		Number learningRate = getLearningRate();
		List<Number> scalings = getScalings();

		PMMLEncoder encoder = schema.getEncoder();

		Schema segmentSchema = schema.toAnonymousSchema();

		MiningModel locModel = NGBoostUtil.encodeParamModel(0, initParams, baseModels, scalings, learningRate, segmentSchema);

		ContinuousFeature locFeature = NGBoostUtil.encodePredictedParam(NGBoostNames.OUTPUT_LOC, locModel, encoder);

		RegressionModel regressionModel = NGBoostUtil.encodeRegression(locFeature, RegressionModel.NormalizationMethod.EXP, schema);

		return MiningModelUtil.createModelChain(Arrays.asList(locModel, regressionModel), Segmentation.MissingPredictionTreatment.RETURN_MISSING);
	}

	public List<List<Regressor>> getBaseModels(){
		return NGBoostUtil.getBaseModels(this);
	}

	public ClassDictConstructor getDist(){
		return get("Dist", ClassDictConstructor.class);
	}

	public String getDistName(){
		Function<String, String> function = new Function<String, String>(){

			@Override
			public String apply(String name){
				String distName = NGBRegressor.this.getString(name);

				// XXX
				if(Objects.equals("DistWithUncensoredScore", distName)){
					ClassDictConstructor dictConstructor = getDist();

					return ClassDictConstructorUtil.getName(dictConstructor);
				}

				return distName;
			}
		};

		return getEnum("Dist_name", function, Arrays.asList(NGBRegressor.DIST_LOGNORMAL, NGBRegressor.DIST_NORMAL, NGBRegressor.DIST_POISSON));
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
	private Expression encodeConfidenceLevel(Object confidenceLevel){

		if(confidenceLevel instanceof Boolean){
			Boolean booleanValue = (Boolean)confidenceLevel;

			return booleanValue ? new FieldRef(NGBoostNames.INPUT_CI) : null;
		} else

		if(confidenceLevel instanceof Number){
			Number number = (Number)confidenceLevel;

			if(!(number.doubleValue() > 0d && number.doubleValue() < 1d)){
				throw new SkLearnException("Expected 0 < ci < 1 confidence level value, got " + number);
			}

			return ExpressionUtil.createConstant(number);
		} else

		if(confidenceLevel instanceof String){
			String string = (String)confidenceLevel;

			return new FieldRef(string);
		} else

		{
			throw new SkLearnException("Expected boolean, double or string confidence level value, got " + ClassDictUtil.formatClass(confidenceLevel));
		}
	}

	static
	private Output encodeBoundsOutput(ContinuousFeature locFeature, ContinuousFeature scaleFeature, Expression confidenceLevelExpression, String function, Schema schema){
		PMMLEncoder encoder = schema.getEncoder();
		ContinuousLabel continuousLabel = schema.requireContinuousLabel();

		if(confidenceLevelExpression instanceof FieldRef){
			FieldRef fieldRef = (FieldRef)confidenceLevelExpression;

			@SuppressWarnings("unused")
			DataField dataField = encoder.createDataField(fieldRef.requireField(), OpType.CONTINUOUS, DataType.DOUBLE)
				.setDisplayName("Confidence level")
				.addIntervals(new Interval(Interval.Closure.OPEN_OPEN, 0, 1));
		}

		DefineFunction defineFunction = encoder.getDefineFunction("zCritical");
		if(defineFunction == null){
			defineFunction = encodeZCriticalFunction("zCritical");

			encoder.addDefineFunction(defineFunction);
		}

		// XXX: Shared between two expressions
		Apply apply = ExpressionUtil.createApply(PMMLFunctions.MULTIPLY,
			ExpressionUtil.createApply(PMMLFunctions.EXP, scaleFeature.ref()),
			ExpressionUtil.createApply(defineFunction, confidenceLevelExpression)
		);

		Apply lowerBoundApply = ExpressionUtil.createApply(PMMLFunctions.SUBTRACT, locFeature.ref(), apply);
		Apply upperBoundApply = ExpressionUtil.createApply(PMMLFunctions.ADD, locFeature.ref(), apply);

		if(function != null){
			lowerBoundApply = ExpressionUtil.createApply(function, lowerBoundApply);
			upperBoundApply = ExpressionUtil.createApply(function, upperBoundApply);
		} // End if

		if(confidenceLevelExpression instanceof FieldRef){
			FieldRef fieldRef = (FieldRef)confidenceLevelExpression;

			lowerBoundApply = encodeIsNotMissingCheck(fieldRef, lowerBoundApply);
			upperBoundApply = encodeIsNotMissingCheck(fieldRef, upperBoundApply);
		}

		OutputField lowerBoundOutputField = new OutputField(FieldNameUtil.create("lower", continuousLabel.getName()), OpType.CONTINUOUS, DataType.DOUBLE)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(lowerBoundApply);

		OutputField upperBoundOutputField = new OutputField(FieldNameUtil.create("upper", continuousLabel.getName()), OpType.CONTINUOUS, DataType.DOUBLE)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(upperBoundApply);

		Output output = new Output()
			.addOutputFields(lowerBoundOutputField, upperBoundOutputField);

		return output;
	}

	static
	private DefineFunction encodeZCriticalFunction(String name){
		ParameterField valueField = new ParameterField("p")
			.setOpType(OpType.CONTINUOUS)
			.setDataType(DataType.DOUBLE);

		Apply apply = ExpressionUtil.createApply(PMMLFunctions.STDNORMALIDF,
			ExpressionUtil.createApply(PMMLFunctions.ADD,
				ExpressionUtil.createConstant(0.5d),
				ExpressionUtil.createApply(PMMLFunctions.DIVIDE, new FieldRef(valueField), ExpressionUtil.createConstant(2d))
			)
		);

		DefineFunction defineFunction = new DefineFunction(name, OpType.CONTINUOUS, DataType.DOUBLE, null, apply)
			.addParameterFields(valueField);

		return defineFunction;
	}

	static
	private Apply encodeIsNotMissingCheck(FieldRef fieldRef, Apply apply){
		return ExpressionUtil.createApply(PMMLFunctions.IF,
			ExpressionUtil.createApply(PMMLFunctions.ISNOTMISSING, fieldRef),
			apply
		);
	}

	private static final String DIST_LOGNORMAL = "LogNormal";
	private static final String DIST_NORMAL = "Normal";
	private static final String DIST_POISSON = "Poisson";
}