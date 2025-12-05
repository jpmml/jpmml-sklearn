/*
 * Copyright (c) 2024 Villu Ruusmann
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
package interpret.glassbox;

import java.util.List;

import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import sklearn.Classifier;

public class GlassboxClassifier extends Classifier {

	public GlassboxClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public List<?> getClasses(){
		return super.getClasses();
	}

	@Override
	public boolean hasProbabilityDistribution(){
		Classifier classifier = getSkModel();

		return classifier.hasProbabilityDistribution();
	}

	@Override
	public Model encodeModel(Schema schema){
		Classifier skModel = getSkModel();

		return skModel.encodeModel(schema);
	}

	public Classifier getSkModel(){
		return getClassifier("sk_model_");
	}
}