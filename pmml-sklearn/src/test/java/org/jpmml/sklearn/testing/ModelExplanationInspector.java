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
package org.jpmml.sklearn.testing;

import java.lang.reflect.Field;

import org.dmg.pmml.Model;
import org.dmg.pmml.ModelExplanation;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.Segment;
import org.jpmml.model.MisplacedElementException;
import org.jpmml.model.MissingElementException;
import org.jpmml.model.visitors.AbstractVisitor;

public class ModelExplanationInspector extends AbstractVisitor {

	@Override
	public VisitorAction visit(Model model){
		PMMLObject parent = getParent();

		if(parent instanceof PMML){
			ensureModelExplanationDefined(model);
		} else

		if(parent instanceof Segment){
			ensureModelExplanationNotDefined(model);
		}

		return super.visit(model);
	}

	private void ensureModelExplanationDefined(Model model){
		ModelExplanation modelExplanation = model.getModelExplanation();

		if(modelExplanation == null){
			Field modelExplanationField;

			try {
				modelExplanationField = (model.getClass()).getDeclaredField("modelExplanation");
			} catch(ReflectiveOperationException roe){
				throw new RuntimeException(roe);
			}

			throw new MissingElementException(model, modelExplanationField);
		}
	}

	private void ensureModelExplanationNotDefined(Model model){
		ModelExplanation modelExplanation = model.getModelExplanation();

		if(modelExplanation != null){
			throw new MisplacedElementException(modelExplanation);
		}
	}
}