/*
 * Copyright (c) 2026 Villu Ruusmann
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
package flaml.automl;

import java.util.Collections;
import java.util.List;

import sklearn.Composite;
import sklearn.Estimator;
import sklearn.Transformer;
import sklearn.preprocessing.Normalizer;

public class SGDEstimator extends Composite {

	public SGDEstimator(String module, String name){
		super(module, name);
	}

	@Override
	public boolean hasTransformers(){
		return true;
	}

	@Override
	public boolean hasFinalEstimator(){
		return true;
	}

	@Override
	public List<? extends Transformer> getTransformers(){
		Normalizer normalizer = getNormalizer();

		return Collections.singletonList(normalizer);
	}

	@Override
	public Estimator getFinalEstimator(){
		return getModel(Estimator.class);
	}

	@Override
	public <E extends Estimator> E getFinalEstimator(Class<? extends E> clazz){
		return getModel(clazz);
	}

	@Override
	public Normalizer getHead(){
		return getNormalizer();
	}

	public Normalizer getNormalizer(){
		return getTransformer("normalizer", Normalizer.class);
	}

	public Estimator getModel(){
		return getModel(Estimator.class);
	}

	public <E extends Estimator> E getModel(Class<? extends E> clazz){
		return getEstimator("_model", clazz);
	}
}