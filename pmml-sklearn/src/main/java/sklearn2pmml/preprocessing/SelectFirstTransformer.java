/*
 * Copyright (c) 2023 Villu Ruusmann
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
package sklearn2pmml.preprocessing;

import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.Scope;
import org.jpmml.python.TupleUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Transformer;
import sklearn2pmml.util.EvaluatableUtil;

public class SelectFirstTransformer extends Transformer {

	public SelectFirstTransformer(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		Transformer controller = getController();
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException();
		}

		ClassDictUtil.checkSize(1, features);

		Feature feature = Iterables.getOnlyElement(features);

		List<Feature> controlFeatures = features;

		if(controller != null){
			controlFeatures = controller.encode(controlFeatures, encoder);
		}

		Scope scope = new DataFrameScope("X", controlFeatures);

		Apply apply = null;

		Apply prevIfApply = null;

		Set<DataType> dataTypes = EnumSet.noneOf(DataType.class);

		for(int i = 0; i < steps.size(); i++){
			Object[] step = steps.get(i);

			String name = TupleUtil.extractElement(step, 0, String.class);
			Transformer transformer = TupleUtil.extractElement(step, 1, Transformer.class);
			Object expr = TupleUtil.extractElement(step, 2, Object.class);

			Apply ifApply = PMMLUtil.createApply(PMMLFunctions.IF, EvaluatableUtil.translateExpression(expr, scope));

			List<Feature> stepFeatures = transformer.encode(Collections.singletonList(feature), encoder);

			ClassDictUtil.checkSize(1, stepFeatures);

			Feature stepFeature = Iterables.getOnlyElement(stepFeatures);

			ifApply.addExpressions(stepFeature.ref());

			dataTypes.add(stepFeature.getDataType());

			if(apply == null){
				apply = ifApply;
			} // End if

			if(prevIfApply != null){
				prevIfApply.addExpressions(ifApply);
			}

			prevIfApply = ifApply;
		}

		DataType dataType = Iterables.getOnlyElement(dataTypes);
		OpType opType = TypeUtil.getOpType(dataType);

		DerivedField derivedField = encoder.createDerivedField(createFieldName("selectFirst", feature), opType, dataType, apply);

		return Collections.singletonList(FeatureUtil.createFeature(derivedField, encoder));
	}

	public Transformer getController(){
		return getOptional("controller", Transformer.class);
	}

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}
}