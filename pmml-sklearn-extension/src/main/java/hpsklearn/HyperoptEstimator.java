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
package hpsklearn;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.converter.Schema;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn.Composite;
import sklearn.Estimator;
import sklearn.Step;
import sklearn.StepUtil;
import sklearn.Transformer;

public class HyperoptEstimator extends Composite implements Encodable {

	public HyperoptEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public boolean hasTransformers(){
		Object[] bestPreprocs = getBestPreprocs();

		return (bestPreprocs.length > 0);
	}

	@Override
	public boolean hasFinalEstimator(){
		return true;
	}

	@Override
	public List<? extends Transformer> getTransformers(){
		Object[] bestPreprocs = getBestPreprocs();

		CastFunction<Transformer> castFunction = new CastFunction<Transformer>(Transformer.class){

			@Override
			public String formatMessage(Object object){
				return "The pre-processor object (" + ClassDictUtil.formatClass(object) + ") is not a supported Transformer";
			}
		};

		return Arrays.stream(bestPreprocs)
			.map(castFunction)
			.collect(Collectors.toList());
	}

	@Override
	public Estimator getFinalEstimator(){
		return getFinalEstimator(Estimator.class);
	}

	@Override
	public <E extends Estimator> E getFinalEstimator(Class<? extends E> clazz){
		E bestLearner = getBestLearner(clazz);

		return bestLearner;
	}

	@Override
	public Step getHead(){
		List<? extends Transformer> transformers = getTransformers();

		if(!transformers.isEmpty()){
			Transformer transformer = transformers.get(0);

			return StepUtil.getHead(transformer);
		}

		Estimator estimator = getFinalEstimator();

		return StepUtil.getHead(estimator);
	}

	@Override
	public PMML encodePMML(){
		SkLearnEncoder encoder = new SkLearnEncoder();

		Estimator estimator = getFinalEstimator();

		initLabel(estimator, null, encoder);
		initFeatures(estimator, null, encoder);

		Schema schema = encoder.createSchema();

		Model model = estimator.encode(schema);

		encoder.setModel(model);

		return encoder.encodePMML(model);
	}

	public Estimator getBestLearner(){
		return getBestLearner(Estimator.class);
	}

	public <E extends Estimator> E getBestLearner(Class<? extends E> clazz){
		return get("_best_learner", clazz);
	}

	public Object[] getBestPreprocs(){
		return getTuple("_best_preprocs");
	}
}