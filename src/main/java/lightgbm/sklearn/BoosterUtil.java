/*
 * Copyright (c) 2017 Villu Ruusmann
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
package lightgbm.sklearn;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.OpType;
import org.dmg.pmml.TypeDefinitionField;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.lightgbm.GBDT;
import org.jpmml.sklearn.SkLearnEncoder;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public int getNumberOfFeatures(HasBooster hasBooster){
		Booster booster = hasBooster.getBooster();

		GBDT gbdt = booster.getGBDT();

		String[] featureNames;

		try {
			Field field = GBDT.class.getDeclaredField("feature_names_");
			field.setAccessible(true);

			featureNames = (String[])field.get(gbdt);
		} catch(Exception e){
			throw new RuntimeException(e);
		}

		return featureNames.length;
	}

	static
	public MiningModel encodeModel(HasBooster hasBooster, Schema schema){
		Booster booster = hasBooster.getBooster();

		GBDT gbdt = booster.getGBDT();

		List<Feature> lgbmFeatures = new ArrayList<>();

		List<Feature> features = schema.getFeatures();
		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Set<Double> featureCategories;

			try {
				Method method = GBDT.class.getDeclaredMethod("getFeatureCategories", int.class);
				method.setAccessible(true);

				featureCategories = (Set<Double>)method.invoke(gbdt, i);
			} catch(Exception e){
				throw new RuntimeException(e);
			}

			OpType opType;

			if(featureCategories != null && featureCategories.size() > 0){
				opType = OpType.CATEGORICAL;
			} else

			{
				opType = OpType.CONTINUOUS;
			}

			SkLearnEncoder encoder = (SkLearnEncoder)continuousFeature.getEncoder();

			TypeDefinitionField field = encoder.getField(continuousFeature.getName());
			field.setOpType(opType);

			lgbmFeatures.add(continuousFeature);
		}

		Schema lgbmSchema = new Schema(schema.getLabel(), lgbmFeatures);

		MiningModel miningModel = gbdt.encodeMiningModel(lgbmSchema);

		return miningModel;
	}
}