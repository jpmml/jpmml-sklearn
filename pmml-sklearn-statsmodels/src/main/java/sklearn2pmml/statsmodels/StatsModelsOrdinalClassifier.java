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
package sklearn2pmml.statsmodels;

import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.Schema;
import org.jpmml.statsmodels.OrdinalLabel;

public class StatsModelsOrdinalClassifier extends StatsModelsClassifier {

	public StatsModelsOrdinalClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		ModelEncoder encoder = (ModelEncoder)schema.getEncoder();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(!categoricalLabel.isAnonymous()){
			DataField dataField = encoder.getDataField(categoricalLabel.getName());

			dataField.setOpType(OpType.ORDINAL);
		}

		OrdinalLabel ordinalLabel = new OrdinalLabel(categoricalLabel.getName(), categoricalLabel.getDataType(), categoricalLabel.getValues());

		schema = schema.toRelabeledSchema(ordinalLabel);

		return super.encodeModel(schema);
	}
}