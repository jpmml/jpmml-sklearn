/*
 * Copyright (c) 2025 Villu Ruusmann
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
package sklearn.preprocessing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.LinearNorm;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutlierTreatmentMethod;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.python.FunctionUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.SkLearnTransformer;

public class QuantileTransformer extends SkLearnTransformer {

	public QuantileTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		int[] shape = getQuantilesShape();

		int numberOfQuantiles = shape[0];
		int numberOfFeatures = shape[1];

		String outputDistribution = getOutputDistribution();
		List<Number> quantiles = getQuantiles();
		List<Number> references = getReferences();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			List<Number> featureQuantiles = CMatrixUtil.getColumn(quantiles, numberOfQuantiles, numberOfFeatures, i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Expression expression = encodeQuantileTransform(continuousFeature, featureQuantiles, references);

			switch(outputDistribution){
				case QuantileTransformer.OUTPUTDISTRIBUTION_UNIFORM:
					break;
				case QuantileTransformer.OUTPUTDISTRIBUTION_NORMAL:
					{
						String name = "clipQuantile";

						DefineFunction defineFunction = encoder.getDefineFunction(name);
						if(defineFunction == null){
							defineFunction = encodeDefineFunction(name, encoder);

							encoder.addDefineFunction(defineFunction);
						}

						expression = ExpressionUtil.createApply(PMMLFunctions.STDNORMALIDF)
							.addExpressions(ExpressionUtil.createApply(defineFunction, expression));
					}
					break;
				default:
					throw new IllegalArgumentException();
			}

			DerivedField derivedField = encoder.createDerivedField(createFieldName("quantile", continuousFeature.getName()), OpType.CONTINUOUS, DataType.DOUBLE, expression);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public String getOutputDistribution(){
		return getEnum("output_distribution", this::getString, Arrays.asList(QuantileTransformer.OUTPUTDISTRIBUTION_UNIFORM, QuantileTransformer.OUTPUTDISTRIBUTION_NORMAL));
	}

	public List<Number> getQuantiles(){
		return getNumberArray("quantiles_");
	}

	public int[] getQuantilesShape(){
		return getArrayShape("quantiles_", 2);
	}

	public List<Number> getReferences(){
		return getNumberArray("references_");
	}

	static
	private NormContinuous encodeQuantileTransform(Feature feature, List<Number> quantiles, List<Number> references){
		NormContinuous normContinupus = new NormContinuous(feature.getName(), null)
			.setOutlierTreatment(OutlierTreatmentMethod.AS_EXTREME_VALUES);

		for(int i = 0; i < quantiles.size(); i++){
			Number orig = quantiles.get(i);
			Number norm = references.get(i);

			normContinupus.addLinearNorms(new LinearNorm(orig, norm));
		}

		return normContinupus;
	}

	static
	private DefineFunction encodeDefineFunction(String name, SkLearnEncoder encoder){
		ParameterField valueField = new ParameterField("x");

		double boundsThreshold = 1e-7;

		Constant lowerBound = ExpressionUtil.createConstant(0d + boundsThreshold);
		Constant upperBound = ExpressionUtil.createConstant(1d - boundsThreshold);

		Apply apply = (Apply)FunctionUtil.encodeFunction("numpy", "clip", Arrays.asList(new FieldRef(valueField), lowerBound, upperBound), encoder);

		DefineFunction defineFunction = new DefineFunction(name, OpType.CONTINUOUS, DataType.DOUBLE, null, apply)
			.addParameterFields(valueField);

		return defineFunction;
	}

	private static final String OUTPUTDISTRIBUTION_UNIFORM = "uniform";
	private static final String OUTPUTDISTRIBUTION_NORMAL = "normal";
}