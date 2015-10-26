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

import java.util.List;

import org.dmg.pmml.CategoricalPredictor;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.PredictorTerm;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.FieldCollector;

public class RegressionModelFieldCollector extends FieldCollector {

	@Override
	public VisitorAction visit(NumericPredictor numericPredictor){
		addField(numericPredictor.getName());

		return super.visit(numericPredictor);
	}

	@Override
	public VisitorAction visit(CategoricalPredictor categoricalPredictor){
		addField(categoricalPredictor.getName());

		return super.visit(categoricalPredictor);
	}

	@Override
	public VisitorAction visit(PredictorTerm predictorTerm){

		if(predictorTerm.hasFieldRefs()){
			List<FieldRef> fieldRefs = predictorTerm.getFieldRefs();

			for(FieldRef fieldRef : fieldRefs){
				addField(fieldRef.getField());
			}
		}

		return super.visit(predictorTerm);
	}
}