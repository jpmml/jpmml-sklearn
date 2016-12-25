/*
 * Copyright (c) 2016 Villu Ruusmann
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
package sklearn2pmml;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sklearn.ClassDictUtil;
import org.jpmml.sklearn.FeatureMapper;
import org.jpmml.sklearn.TupleUtil;
import sklearn.pipeline.Pipeline;
import sklearn_pandas.DataFrameMapper;

public class PMMLPipeline extends Pipeline {

	public PMMLPipeline(String module, String name){
		super(module, name);
	}

	@Override
	public Schema createSchema(FeatureMapper featureMapper){

		if(isSupervised()){
			FieldName targetField = createTargetField();

			DataField dataField = featureMapper.createDataField(targetField);

			Feature feature = new WildcardFeature(dataField);

			featureMapper.addRow(Collections.singletonList(feature));
		}

		return super.createSchema(featureMapper);
	}

	public PMML encodePMML(){
		DataFrameMapper dataFrameMapper = getMapper();

		FeatureMapper featureMapper = new FeatureMapper();

		if(dataFrameMapper != null){
			dataFrameMapper.encodeFeatures(featureMapper);
		}

		Set<DefineFunction> defineFunctions = encodeDefineFunctions();
		for(DefineFunction defineFunction : defineFunctions){
			featureMapper.addDefineFunction(defineFunction);
		}

		Schema schema = createSchema(featureMapper);

		Model model = encodeModel(schema);

		return featureMapper.encodePMML(model);
	}

	public DataFrameMapper getMapper(){
		Object[] mapperStep = getMapperStep();

		if(mapperStep != null){
			return (DataFrameMapper)TupleUtil.extractElement(mapperStep, 1);
		}

		return null;
	}

	public Object[] getMapperStep(){
		List<Object[]> selectorSteps = super.getSelectorSteps();

		if(selectorSteps.size() > 0){
			Object object = TupleUtil.extractElement(selectorSteps.get(0), 1);

			if(object instanceof DataFrameMapper){
				return selectorSteps.get(0);
			}
		}

		return null;
	}

	@Override
	public List<Object[]> getSelectorSteps(){
		List<Object[]> selectorSteps = super.getSelectorSteps();

		if(selectorSteps.size() > 0){
			Object object = TupleUtil.extractElement(selectorSteps.get(0), 1);

			if(object instanceof DataFrameMapper){
				selectorSteps = selectorSteps.subList(1, selectorSteps.size());
			}
		}

		return selectorSteps;
	}

	@Override
	protected FieldName createTargetField(){
		String targetField = getTargetField();

		if(targetField != null){
			return FieldName.create(targetField);
		}

		return super.createTargetField();
	}

	public List<String> getActiveFields(){
		return (List)ClassDictUtil.getArray(this, "active_fields");
	}

	public String getTargetField(){
		return (String)get("target_field");
	}
}