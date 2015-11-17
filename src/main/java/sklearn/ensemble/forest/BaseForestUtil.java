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
package sklearn.ensemble.forest;

import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.TreeModel;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sklearn.Schema;
import org.jpmml.sklearn.SchemaUtil;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn.tree.HasTree;
import sklearn.tree.TreeModelUtil;

public class BaseForestUtil {

	private BaseForestUtil(){
	}

	static
	public <E extends Estimator & HasTree> MiningModel encodeBaseForest(List<E> estimators, MultipleModelMethodType multipleModelMethod, final MiningFunctionType miningFunction, final Schema schema){
		Function<E, TreeModel> function = new Function<E, TreeModel>(){

			private Schema segmentSchema = SchemaUtil.createSegmentSchema(schema);


			@Override
			public TreeModel apply(E estimator){
				return TreeModelUtil.encodeTreeModel(estimator, miningFunction, this.segmentSchema);
			}
		};

		List<TreeModel> treeModels = Lists.transform(estimators, function);

		Segmentation segmentation = EstimatorUtil.encodeSegmentation(multipleModelMethod, treeModels, null);

		MiningSchema miningSchema = PMMLUtil.createMiningSchema(schema.getTargetField(), schema.getActiveFields());

		MiningModel miningModel = new MiningModel(miningFunction, miningSchema)
			.setSegmentation(segmentation);

		return miningModel;
	}
}