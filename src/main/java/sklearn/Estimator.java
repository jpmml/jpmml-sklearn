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
package sklearn;

import java.util.ArrayList;
import java.util.List;

import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.PMMLUtil;

abstract
public class Estimator extends ClassDict {

	public Estimator(String module, String name){
		super(module, name);
	}

	abstract
	public DataField encodeTarget();

	abstract
	public Model encodeModel(List<DataField> dataFields);

	public PMML encodePMML(){
		List<DataField> dataFields = new ArrayList<>();

		DataField targetDataField = encodeTarget();
		dataFields.add(targetDataField);

		Integer features = getFeatures();
		for(int i = 0; i < features.intValue(); i++){
			DataField dataField = new DataField(FieldName.create("x" + String.valueOf(i + 1)), OpType.CONTINUOUS, DataType.DOUBLE);

			dataFields.add(dataField);
		}

		DataDictionary dataDictionary = new DataDictionary(dataFields);

		PMML pmml = new PMML("4.2", PMMLUtil.createHeader("JPMML-SkLearn"), dataDictionary);

		Model model = encodeModel(dataFields);

		pmml.addModels(model);

		return pmml;
	}

	public Integer getFeatures(){
		return (Integer)get("n_features_");
	}
}