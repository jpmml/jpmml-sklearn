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
package sklearn.ensemble.bagging;

import java.util.ArrayList;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import numpy.core.NDArray;
import numpy.core.NDArrayUtil;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Model;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Segmentation;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.Schema;
import sklearn.EstimatorUtil;
import sklearn.Regressor;

public class BaggingRegressor extends Regressor {

	public BaggingRegressor(String module, String name){
		super(module, name);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<Regressor> estimators = getEstimators();
		List<List<? extends Number>> estimatorsFeatures = getEstimatorsFeatures();

		List<Model> models = new ArrayList<>();

		for(int i = 0; i < estimators.size(); i++){
			Regressor estimator = estimators.get(i);
			List<? extends Number> estimatorFeatures = estimatorsFeatures.get(i);

			List<FieldName> activeFields = new ArrayList<>();

			for(Number estimatorFeature : estimatorFeatures){
				FieldName activeField = schema.getActiveField(estimatorFeature.intValue());

				activeFields.add(activeField);
			}

			Schema estimatorSchema = new Schema(null, activeFields);

			Model model = estimator.encodeModel(estimatorSchema);

			models.add(model);
		}

		Segmentation segmentation = EstimatorUtil.encodeSegmentation(MultipleModelMethodType.AVERAGE, models, null);

		MiningSchema miningSchema = PMMLUtil.createMiningSchema(schema.getTargetField(), schema.getActiveFields());

		MiningModel miningModel = new MiningModel(MiningFunctionType.REGRESSION, miningSchema)
			.setSegmentation(segmentation);

		return miningModel;
	}

	public List<Regressor> getEstimators(){
		Function<Object, Regressor> function = new Function<Object, Regressor>(){

			@Override
			public Regressor apply(Object object){

				if(object instanceof Regressor){
					return (Regressor)object;
				}

				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not a Regressor or is not a supported Regressor subclass");
			}
		};

		return Lists.transform((List)get("estimators_"), function);
	}

	public List<List<? extends Number>> getEstimatorsFeatures(){
		Function<Object, List<? extends Number>> function = new Function<Object, List<? extends Number>>(){

			@Override
			public List<? extends Number> apply(Object object){
				object = NDArrayUtil.unwrap(object);

				if(object instanceof NDArray){
					NDArray array = (NDArray)object;

					return (List)array.getContent();
				}

				throw new IllegalArgumentException("The estimator features object (" + ClassDictUtil.formatClass(object) + ") is not an array");
			}
		};

		return Lists.transform((List)get("estimators_features_"), function);
	}
}