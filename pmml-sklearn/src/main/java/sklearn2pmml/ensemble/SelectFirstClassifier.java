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
import java.util.Objects;

import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.python.TupleUtil;
import sklearn.Classifier;

public class SelectFirstClassifier extends Classifier implements HasEstimatorSteps {

	public SelectFirstClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		List<? extends Classifier> classifiers = getClassifiers();

		List<?> result = null;

		for(Classifier classifier : classifiers){

			if(result == null){
				result = classifier.getClasses();
			} else

			{
				if(!Objects.equals(result, classifier.getClasses())){
					throw new IllegalArgumentException();
				}
			}
		}

		return result;
	}

	@Override
	public boolean hasProbabilityDistribution(){
		List<? extends Classifier> classifiers = getClassifiers();

		boolean result = true;

		for(Classifier classifier : classifiers){
			result &= classifier.hasProbabilityDistribution();
		}

		return result;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		return SelectFirstUtil.encodeClassifier(this, schema);
	}

	public List<? extends Classifier> getClassifiers(){
		List<Object[]> steps = getSteps();

		if(steps.isEmpty()){
			throw new IllegalArgumentException();
		}

		return TupleUtil.extractElementList(steps, 1, Classifier.class);
	}

	@Override
	public List<Object[]> getSteps(){
		return getTupleList("steps");
	}
}