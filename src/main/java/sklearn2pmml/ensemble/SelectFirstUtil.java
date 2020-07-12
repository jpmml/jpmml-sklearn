/*
 * Copyright (c) 2020 Villu Ruusmann
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
package sklearn2pmml.ensemble;

import java.util.List;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.python.DataFrameScope;
import org.jpmml.python.PredicateTranslator;
import org.jpmml.python.Scope;
import org.jpmml.python.TupleUtil;
import sklearn.Estimator;

public class SelectFirstUtil {

	private SelectFirstUtil(){
	}

	static
	public MiningModel encodeRegressor(List<Object[]> steps, Schema schema){
		return encodeModel(MiningFunction.REGRESSION, steps, schema);
	}

	static
	public MiningModel encodeClassifier(List<Object[]> steps, Schema schema){
		return encodeModel(MiningFunction.CLASSIFICATION, steps, schema);
	}

	static
	private MiningModel encodeModel(MiningFunction miningFunction, List<Object[]> steps, Schema schema){

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		Segmentation segmentation = new Segmentation(Segmentation.MultipleModelMethod.SELECT_FIRST, null);

		Scope scope = new DataFrameScope(FieldName.create("X"), features);

		for(Object[] step : steps){
			String name = TupleUtil.extractElement(step, 0, String.class);
			Estimator estimator = TupleUtil.extractElement(step, 1, Estimator.class);
			String predicate = TupleUtil.extractElement(step, 2, String.class);

			if(!(miningFunction).equals(estimator.getMiningFunction())){
				throw new IllegalArgumentException();
			}

			Predicate pmmlPredicate = PredicateTranslator.translate(predicate, scope);

			Model model = estimator.encode(schema);

			Segment segment = new Segment(pmmlPredicate, model)
				.setId(name);

			segmentation.addSegments(segment);
		}

		MiningModel miningModel = new MiningModel(miningFunction, ModelUtil.createMiningSchema(label))
			.setSegmentation(segmentation);

		return miningModel;
	}
}