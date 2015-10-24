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
package sklearn.ensemble;

import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Output;
import org.jpmml.converter.PMMLUtil;
import sklearn.Classifier;
import sklearn.tree.DecisionTreeClassifier;
import sklearn.tree.TreeModelUtil;

public class RandomForestClassifier extends Classifier {

	public RandomForestClassifier(String module, String name){
		super(module, name);
	}

	@Override
	public DataType getDataType(){
		return DataType.FLOAT;
	}

	@Override
	public MiningModel encodeModel(List<DataField> dataFields){
		DataField dataField = dataFields.get(0);

		Output output = new Output(PMMLUtil.createProbabilityFields(dataField));

		List<DecisionTreeClassifier> estimators = getEstimators();

		MiningModel miningModel = TreeModelUtil.encodeTreeModelEnsemble(estimators, null, MultipleModelMethodType.AVERAGE, MiningFunctionType.CLASSIFICATION, dataFields, true)
			.setOutput(output);

		return miningModel;
	}

	public List<DecisionTreeClassifier> getEstimators(){
		return (List)get("estimators_");
	}
}