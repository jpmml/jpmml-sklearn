/*
 * Copyright (c) 2020 Villu Ruusmann
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
package org.jpmml.sklearn.visitors;

import java.lang.reflect.Method;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.Model;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.visitors.DeepFieldResolver;
import org.jpmml.converter.visitors.FieldDependencyResolver;
import org.jpmml.converter.visitors.FieldUtil;

public class FeatureExpander extends DeepFieldResolver {

	private Map<Model, Set<FieldName>> features = null;

	private Map<Model, Map<FieldName, Set<Field<?>>>> expandedFeatures = new IdentityHashMap<>();


	public FeatureExpander(Map<Model, Set<FieldName>> features){
		this.features = Objects.requireNonNull(features);
	}

	@Override
	public void reset(){
		super.reset();

		this.features.clear();
		this.expandedFeatures.clear();
	}

	@Override
	public PMMLObject popParent(){
		PMMLObject parent = super.popParent();

		if(parent instanceof Model){
			Model model = (Model)parent;

			processModel(model);
		}

		return parent;
	}

	private void processModel(Model model){
		FieldDependencyResolver fieldDependencyResolver = getFieldDependencyResolver();

		MiningModel parentMiningModel = null;

		Set<FieldName> features = this.features.get(model);
		if(features == null){
			parentMiningModel = getParent(this.features.keySet());

			if(parentMiningModel != null){
				features = this.features.get(parentMiningModel);
			} // End if

			if(features == null){
				return;
			}
		}

		Collection<Field<?>> modelFields = getFields(model);

		Collection<Field<?>> featureFields = FieldUtil.selectAll(modelFields, features, true);

		Map<FieldName, DerivedField> localDerivedFields = Collections.emptyMap();

		LocalTransformations localTransformations = model.getLocalTransformations();
		if(localTransformations != null && localTransformations.hasDerivedFields()){
			localDerivedFields = FieldUtil.nameMap(localTransformations.getDerivedFields());
		} // End if

		if(parentMiningModel != null){

			if(localDerivedFields.isEmpty()){
				return;
			}

			featureFields.retainAll(localDerivedFields.values());
		}

		Map<FieldName, DerivedField> globalDerivedFields;

		try {
			Method method = FieldDependencyResolver.class.getDeclaredMethod("getGlobalDerivedFields");

			if(!method.isAccessible()){
				method.setAccessible(true);
			}

			globalDerivedFields = FieldUtil.nameMap((Collection)method.invoke(fieldDependencyResolver));
		} catch(ReflectiveOperationException roe){
			throw new IllegalArgumentException(roe);
		}

		Map<FieldName, Set<Field<?>>> expandedFields;

		if(parentMiningModel != null){
			expandedFields = ensureExpandedFeatures(parentMiningModel);
		} else

		{
			expandedFields = ensureExpandedFeatures(model);
		}

		for(Field<?> featureField : featureFields){
			FieldName name = featureField.getName();

			if(featureField instanceof DataField){
				expandedFields.put(name, Collections.singleton(featureField));
			} else

			if(featureField instanceof DerivedField){
				DerivedField derivedField = (DerivedField)featureField;

				Set<Field<?>> expandedFeatureFields = new HashSet<>();
				expandedFeatureFields.add(derivedField);

				fieldDependencyResolver.expand(expandedFeatureFields, new HashSet<>(localDerivedFields.values()));
				fieldDependencyResolver.expand(expandedFeatureFields, new HashSet<>(globalDerivedFields.values()));

				expandedFields.put(name, expandedFeatureFields);
			} else

			if(featureField instanceof OutputField){
				expandedFields.put(name, Collections.singleton(featureField));
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	private MiningModel getParent(Set<Model> models){
		Deque<PMMLObject> parents = getParents();

		for(PMMLObject parent : parents){

			if(parent instanceof MiningModel){
				MiningModel miningModel = (MiningModel)parent;

				if(models.contains(miningModel)){
					return miningModel;
				}
			}
		}

		return null;
	}

	private Map<FieldName, Set<Field<?>>> ensureExpandedFeatures(Model model){
		Map<Model, Map<FieldName, Set<Field<?>>>> expandedFeatures = getExpandedFeatures();

		Map<FieldName, Set<Field<?>>> result = expandedFeatures.get(model);
		if(result == null){
			result = new HashMap<>();

			expandedFeatures.put(model, result);
		}

		return result;
	}

	public Map<FieldName, Set<Field<?>>> getExpandedFeatures(Model model){
		Map<Model, Map<FieldName, Set<Field<?>>>> expandedFeatures = getExpandedFeatures();

		return expandedFeatures.get(model);
	}

	public Map<Model, Set<FieldName>> getFeatures(){
		return this.features;
	}

	public Map<Model, Map<FieldName, Set<Field<?>>>> getExpandedFeatures(){
		return this.expandedFeatures;
	}
}