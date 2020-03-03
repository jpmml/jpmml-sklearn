/*
 * Copyright (c) 2019 Villu Ruusmann
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
import org.jpmml.sklearn.PredicateTranslator;
import org.jpmml.sklearn.TupleUtil;
import sklearn.ClassifierUtil;
import sklearn.Estimator;
import sklearn.HasClasses;

public class SelectFirstEstimator extends Estimator implements HasClasses {

	public SelectFirstEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public MiningFunction getMiningFunction(){
		Estimator estimator = getEstimator();

		return estimator.getMiningFunction();
	}

	@Override
	public List<?> getClasses(){
		Estimator estimator = getEstimator();

		return ClassifierUtil.getClasses(estimator);
	}

	@Override
	public Model encodeModel(Schema schema){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		MiningFunction miningFunction = null;

		Segmentation segmentation = new Segmentation(Segmentation.MultipleModelMethod.SELECT_FIRST, null);

		for(Object[] step : steps){
			String name;
			Estimator estimator;
			String predicate;

			// SkLearn2PMML 0.54.0
			if(step.length == 2){
				predicate = TupleUtil.extractElement(step, 0, String.class);
				estimator = TupleUtil.extractElement(step, 1, Estimator.class);
			} else

			// SkLearn2PMML 0.55.0+
			if(step.length == 3){
				name = TupleUtil.extractElement(step, 0, String.class);
				estimator = TupleUtil.extractElement(step, 1, Estimator.class);
				predicate = TupleUtil.extractElement(step, 2, String.class);
			} else

			{
				throw new IllegalArgumentException();
			} // End if

			if(miningFunction == null){
				miningFunction = estimator.getMiningFunction();
			} else

			{
				if(!(miningFunction).equals(estimator.getMiningFunction())){
					throw new IllegalArgumentException();
				}
			}

			Predicate pmmlPredicate = PredicateTranslator.translate(predicate, features);

			Model model = estimator.encodeModel(schema);

			Segment segment = new Segment(pmmlPredicate, model);

			segmentation.addSegments(segment);
		}

		MiningModel miningModel = new MiningModel(miningFunction, ModelUtil.createMiningSchema(label))
			.setSegmentation(segmentation);

		return miningModel;
	}

	public Estimator getEstimator(){
		List<Object[]> steps = getSteps();

		if(steps.size() < 1){
			throw new IllegalArgumentException();
		}

		Object[] step = steps.get(0);

		return TupleUtil.extractElement(step, 1, Estimator.class);
	}

	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}
}