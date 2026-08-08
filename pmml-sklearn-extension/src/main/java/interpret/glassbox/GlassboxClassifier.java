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
import sklearn.Step;

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
		Classifier skModel = getSkModel();

		return skModel.hasProbabilityDistribution();
	}

	@Override
	public Model encode(Step parent, Schema schema){
		Classifier skModel = getSkModel();

		Step prevParent = skModel.getParent();

		try {
			skModel.setParent(this);

			return super.encode(parent, schema);
		} finally {
			skModel.setParent(prevParent);
		}
	}

	@Override
	public Model encodeModel(Schema schema){
		Classifier skModel = getSkModel();

		return skModel.encodeModel(schema);
	}

	@Override
	public Schema configureSchema(Schema schema){
		Classifier skModel = getSkModel();

		return skModel.configureSchema(schema);
	}

	@Override
	public Model configureModel(Model model){
		Classifier skModel = getSkModel();

		return skModel.configureModel(model);
	}

	public Classifier getSkModel(){
		return getClassifier("sk_model_");
	}
}