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
package sklearn.linear_model;

import java.util.ArrayList;
import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.Model;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.MultiLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.python.ClassDictUtil;
import sklearn.Regressor;

public class LinearRegressor extends Regressor {

	public LinearRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public int getNumberOfFeatures(){
		int[] shape = getCoefShape();

		if(shape.length == 2){
			return shape[1];
		}

		return shape[0];
	}

	@Override
	public int getNumberOfOutputs(){
		int[] shape = getCoefShape();

		if(shape.length == 2){
			return shape[0];
		}

		return 1;
	}

	@Override
	public Model encodeModel(Schema schema){
		List<? extends Number> coef = getCoef();
		List<? extends Number> intercept = getIntercept();

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		int numberOfOutputs = getNumberOfOutputs();
		if(numberOfOutputs == 1){
			return createRegression(coef, Iterables.getOnlyElement(intercept), schema);
		} else

		if(numberOfOutputs >= 2){
			MultiLabel multiLabel = (MultiLabel)label;

			ClassDictUtil.checkSize(numberOfOutputs, intercept, multiLabel.getLabels());

			List<Model> models = new ArrayList<>();

			for(int i = 0, max = numberOfOutputs; i < max; i++){
				Schema segmentSchema = schema.toRelabeledSchema(multiLabel.getLabel(i));

				Model model = createRegression(CMatrixUtil.getRow(coef, numberOfOutputs, features.size(), i), intercept.get(i), segmentSchema);

				models.add(model);
			}

			return MiningModelUtil.createMultiModelChain(models, Segmentation.MissingPredictionTreatment.CONTINUE);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	protected RegressionModel createRegression(List<? extends Number> coef, Number intercept, Schema schema){
		return RegressionModelUtil.createRegression(schema.getFeatures(), coef, intercept, null, schema);
	}

	public List<? extends Number> getCoef(){
		return getNumberArray("coef_");
	}

	public int[] getCoefShape(){
		return getArrayShape("coef_");
	}

	public List<? extends Number> getIntercept(){
		return getNumberArray("intercept_");
	}
}