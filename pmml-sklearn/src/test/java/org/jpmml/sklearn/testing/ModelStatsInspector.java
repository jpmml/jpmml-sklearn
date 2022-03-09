/*
 * Copyright (c) 2022 Villu Ruusmann
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
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.Segment;
import org.jpmml.model.MisplacedElementException;
import org.jpmml.model.MissingElementException;
import org.jpmml.model.visitors.AbstractVisitor;

public class ModelStatsInspector extends AbstractVisitor {

	@Override
	public VisitorAction visit(Model model){
		PMMLObject parent = getParent();

		if(parent instanceof PMML){
			ensureModelStatsDefined(model);
		} else

		if(parent instanceof Segment){
			ensureModelStatsNotDefined(model);
		}

		return super.visit(model);
	}

	private void ensureModelStatsDefined(Model model){
		ModelStats modelStats = model.getModelStats();

		if(modelStats == null){
			Field modelStatsField;

			try {
				modelStatsField = (model.getClass()).getDeclaredField("modelStats");
			} catch(ReflectiveOperationException roe){
				throw new RuntimeException(roe);
			}

			throw new MissingElementException(model, modelStatsField);
		}
	}

	private void ensureModelStatsNotDefined(Model model){
		ModelStats modelStats = model.getModelStats();

		if(modelStats != null){
			throw new MisplacedElementException(modelStats);
		}
	}
}