/*
 * Copyright (c) 2021 Villu Ruusmann
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
import java.util.Collections;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import numpy.DType;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Interval;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.HasArray;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;

public class KBinsDiscretizer extends Transformer {

	public KBinsDiscretizer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		DType dtype = getDType();
		String encode = getEncode();
		List<Integer> numberOfBins = getNumberOfBins();
		List<List<Number>> binEdges = getBinEdges();

		ClassDictUtil.checkSize(1, features);

		Feature feature = features.get(0);

		ContinuousFeature continuousFeature = feature.toContinuousFeature();

		continuousFeature = addEps(continuousFeature, encoder);

		List<Number> bins = binEdges.get(0);
		if(bins.isEmpty()){
			throw new IllegalArgumentException();
		}

		ClassDictUtil.checkSize(numberOfBins.get(0) + 1, bins);

		List<Integer> labelCategories = new ArrayList<>();

		Discretize discretize = new Discretize(continuousFeature.getName())
			.setDataType((dtype != null ? dtype.getDataType() : continuousFeature.getDataType()));

		for(int i = 0; i < bins.size() - 1; i++){
			Number leftMargin = (i > 0 ? bins.get(i) : null);
			Number rightMargin = (i < (bins.size() - 1) - 1 ? bins.get(i + 1) : null);

			Interval interval = new Interval(Interval.Closure.CLOSED_OPEN)
				.setLeftMargin(leftMargin)
				.setRightMargin(rightMargin);

			Integer label = i;

			labelCategories.add(label);

			DiscretizeBin discretizeBin = new DiscretizeBin(label, interval);

			discretize.addDiscretizeBins(discretizeBin);
		}

		DerivedField derivedField = encoder.createDerivedField(createFieldName("discretize", continuousFeature), OpType.CATEGORICAL, discretize.getDataType(), discretize);

		switch(encode){
			case "onehot":
			case "onehot-dense":
				{
					List<Feature> result = new ArrayList<>();

					for(int i = 0; i < labelCategories.size(); i++){
						Integer label = labelCategories.get(i);

						result.add(new BinaryFeature(encoder, derivedField, label));
					}

					return result;
				}
			case "ordinal":
				{
					return Collections.singletonList(new IndexFeature(encoder, derivedField, labelCategories));
				}
			default:
				throw new IllegalArgumentException(encode);
		}
	}

	public DType getDType(){
		return (DType)getOptionalDType("dtype", false);
	}

	public String getEncode(){
		return getString("encode");
	}

	public List<Integer> getNumberOfBins(){
		return getIntegerArray("n_bins_");
	}

	public List<List<Number>> getBinEdges(){
		List<? extends HasArray> arrays = getArray("bin_edges_", HasArray.class);

		Function<HasArray, List<Number>> function = new Function<HasArray, List<Number>>(){

			@Override
			public List<Number> apply(HasArray hasArray){
				return (List)hasArray.getArrayContent();
			}
		};

		return Lists.transform(arrays, function);
	}

	static
	private ContinuousFeature addEps(ContinuousFeature continuousFeature, SkLearnEncoder encoder){
		DefineFunction defineFunction = encoder.getDefineFunction("add_eps");
		if(defineFunction == null){
			defineFunction = encodeDefineFunction("add_eps");

			encoder.addDefineFunction(defineFunction);
		}

		Apply apply = PMMLUtil.createApply(defineFunction, continuousFeature.ref());

		DerivedField derivedField = encoder.createDerivedField(FieldNameUtil.create(defineFunction, continuousFeature.getName()), defineFunction.requireOpType(), defineFunction.requireDataType(), apply);

		return new ContinuousFeature(encoder, derivedField);
	}

	static
	private DefineFunction encodeDefineFunction(String name){
		ParameterField valueField = new ParameterField("x");

		Double atol = 1.0e-8;
		Double rtol = 1.0e-5;

		// $name + (atol + rtol * abs($name))
		Apply apply = PMMLUtil.createApply(PMMLFunctions.ADD, new FieldRef(valueField), PMMLUtil.createApply(PMMLFunctions.ADD, PMMLUtil.createConstant(atol), PMMLUtil.createApply(PMMLFunctions.MULTIPLY, PMMLUtil.createConstant(rtol), PMMLUtil.createApply(PMMLFunctions.ABS, new FieldRef(valueField)))));

		DefineFunction defineFunction = new DefineFunction(name, OpType.CONTINUOUS, DataType.DOUBLE, null, apply)
			.addParameterFields(valueField);

		return defineFunction;
	}
}