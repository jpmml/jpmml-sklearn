/*
 * Copyright (c) 2015 Villu Ruusmann
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
package sklearn.decomposition;

import java.util.ArrayList;
import java.util.List;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.SkLearnEncoder;

public class PCA extends BasePCA {

	public PCA(String module, String name){
		super(module, name);
	}

	@Override
	public List<Feature> encodeFeatures(List<Feature> features, SkLearnEncoder encoder){
		int[] shape = getComponentsShape();

		int numberOfComponents = shape[0];
		int numberOfFeatures = shape[1];

		List<? extends Number> components = getComponents();
		List<? extends Number> mean = getMean();

		ClassDictUtil.checkSize(numberOfFeatures, features, mean);

		Boolean whiten = getWhiten();

		List<? extends Number> explainedVariance = (whiten ? getExplainedVariance() : null);

		ClassDictUtil.checkSize(numberOfComponents, explainedVariance);

		FieldName name = createFieldName("pca", features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < numberOfComponents; i++){
			List<? extends Number> component = CMatrixUtil.getRow(components, numberOfComponents, numberOfFeatures, i);

			Apply apply = PMMLUtil.createApply(PMMLFunctions.SUM);

			for(int j = 0; j < numberOfFeatures; j++){
				Feature feature = features.get(j);

				Number meanValue = mean.get(j);
				Number componentValue = component.get(j);

				if(ValueUtil.isZero(meanValue) && ValueUtil.isOne(componentValue)){
					apply.addExpressions(feature.ref());

					continue;
				}

				ContinuousFeature continuousFeature = feature.toContinuousFeature();

				// "($name[i] - mean[i]) * component[i]"
				Expression expression = continuousFeature.ref();

				if(!ValueUtil.isZero(meanValue)){
					expression = PMMLUtil.createApply(PMMLFunctions.SUBTRACT, expression, PMMLUtil.createConstant(meanValue));
				} // End if

				if(!ValueUtil.isOne(componentValue)){
					expression = PMMLUtil.createApply(PMMLFunctions.MULTIPLY, expression, PMMLUtil.createConstant(componentValue));
				}

				apply.addExpressions(expression);
			}

			if(whiten){
				Number explainedVarianceValue = explainedVariance.get(i);

				if(!ValueUtil.isOne(explainedVarianceValue)){
					apply = PMMLUtil.createApply(PMMLFunctions.DIVIDE, apply, PMMLUtil.createConstant(Math.sqrt(ValueUtil.asDouble(explainedVarianceValue))));
				}
			}

			DerivedField derivedField = encoder.createDerivedField(FieldNameUtil.select(name, i), apply);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}

	public Boolean getWhiten(){
		return getBoolean("whiten");
	}

	public List<? extends Number> getExplainedVariance(){
		return getNumberArray("explained_variance_");
	}

	public List<? extends Number> getMean(){
		return getNumberArray("mean_");
	}
}