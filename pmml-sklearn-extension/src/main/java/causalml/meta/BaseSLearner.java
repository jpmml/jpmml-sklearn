/*
 * Copyright (c) 2026 Villu Ruusmann
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
package causalml.meta;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import causalml.meta.visitors.TreeModelGroupActivator;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DiscreteFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.ScalarLabelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.UnsupportedFeatureException;
import org.jpmml.converter.visitors.TreeModelPruner;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;
import sklearn.tree.HasTreeOptions;
import sklearn.tree.TreeUtil;

abstract
public class BaseSLearner<E extends Estimator> extends BaseLearner<E> implements HasTreeOptions {

	public BaseSLearner(String module, String name){
		super(module, name);
	}

	@Override
	public Model encodeModel(Schema schema){
		String controlName = getControlName();
		List<String> treatmentGroups = getTreatmentGroups();
		Map<String, E> models = getModels();

		ModelEncoder encoder = schema.getEncoder();
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		List<ContinuousLabel> continuousLabels = ScalarLabelUtil.toScalarLabels(ContinuousLabel.class, label);

		ClassDictUtil.checkSize(continuousLabels, treatmentGroups, models.entrySet());

		Feature groupFeature = features.get(0);

		if(groupFeature instanceof DiscreteFeature){
			DiscreteFeature binaryFeature = ((DiscreteFeature)groupFeature)
				.expectCardinality(continuousLabels.size() + 1);
		} else

		{
			throw new UnsupportedFeatureException("Expected a categorical feature, got " + groupFeature.typeString());
		}

		String groupName = groupFeature.getName();

		List<Model> binaryModels = new ArrayList<>();

		for(int i = 0; i < continuousLabels.size(); i++){
			ContinuousLabel continuousLabel = continuousLabels.get(i);
			String treatmentGroup = treatmentGroups.get(i);

			E estimator = models.get(treatmentGroup);

			Schema segmentSchema = schema.toRelabeledSchema(continuousLabel);

			BinaryFeature controlFeature = new BinaryFeature(encoder, groupFeature, controlName);

			// XXX
			List<Feature> segmentFeatures = (List<Feature>)segmentSchema.getFeatures();
			segmentFeatures.set(0, controlFeature);

			Model binaryModel = encodeBinaryModel(estimator, groupName, controlName, segmentSchema);

			binaryModels.add(binaryModel);
		}

		return BaseLearnerUtil.encodeModel(binaryModels);
	}

	@Override
	public Schema configureSchema(Schema schema){

		if(hasTreeOptions()){
			Feature controlFeature = schema.getFeature(0);

			Function<Feature, Feature> function = Function.identity();

			Schema treeSchema = schema.toTransformedSchema(function);

			treeSchema = TreeUtil.configureSchema(this, treeSchema);

			// XXX
			List<Feature> treeFeatures = (List<Feature>)treeSchema.getFeatures();
			treeFeatures.set(0, controlFeature);

			return treeSchema;
		}

		return super.configureSchema(schema);
	}

	@Override
	public Model configureModel(Model model){

		if(hasTreeOptions()){
			return TreeUtil.configureModel(this, model);
		}

		return super.configureModel(model);
	}

	protected MiningModel encodeBinaryModel(E estimator, String groupName, String controlName, Schema schema){
		Model controlModel = encodeEstimator(Role.CONTROL, estimator, schema);
		Model treatmentModel = encodeEstimator(Role.TREATMENT, estimator, schema);

		controlModel = optimizeControlModel(controlModel, groupName, controlName);
		treatmentModel = optimizeTreatmentModel(treatmentModel, groupName, controlName);

		return encodeBinaryModel(controlModel, treatmentModel, schema);
	}

	protected MiningModel encodeBinaryModel(Model controlModel, Model treatmentModel, Schema schema){
		return BaseLearnerUtil.encodeBinaryRegressor(controlModel, treatmentModel, schema);
	}

	protected Model optimizeControlModel(Model model, String groupName, String controlName){
		model = preOptimizeModel(model, groupName);

		Visitor controlActivator = new TreeModelGroupActivator(){

			@Override
			public Boolean getActivation(Predicate predicate){

				if(hasFieldReference(predicate, groupName)){

					if(hasOperator(predicate, SimplePredicate.Operator.EQUAL)){
						return !hasValue(predicate, controlName);
					} else

					if(hasOperator(predicate, SimplePredicate.Operator.NOT_EQUAL)){
						return hasValue(predicate, controlName);
					}
				}

				return null;
			}
		};
		controlActivator.applyTo(model);

		model = postOptimizeModel(model, groupName);

		return model;
	}

	protected Model optimizeTreatmentModel(Model model, String groupName, String controlName){
		model = preOptimizeModel(model, groupName);

		Visitor treatmentActivator = new TreeModelGroupActivator(){

			@Override
			public Boolean getActivation(Predicate predicate){

				if(hasFieldReference(predicate, groupName)){

					if(hasOperator(predicate, SimplePredicate.Operator.EQUAL)){
						return hasValue(predicate, controlName);
					} else

					if(hasOperator(predicate, SimplePredicate.Operator.NOT_EQUAL)){
						return !hasValue(predicate, controlName);
					}
				}

				return null;
			}
		};
		treatmentActivator.applyTo(model);

		model = postOptimizeModel(model, groupName);

		return model;
	}

	protected Model preOptimizeModel(Model model, String groupName){
		return model;
	}

	protected Model postOptimizeModel(Model model, String groupName){
		Visitor nodePruner = new TreeModelPruner(){

			private MiningFunction miningFunction = null;


			@Override
			public void exitNode(Node node){
				Object defaultChild = node.getDefaultChild();

				if(node.hasNodes()){
					List<Node> children = node.getNodes();

					if(children.size() == 1){
						Node child = children.get(0);

						Predicate childPredicate = child.requirePredicate();

						if(childPredicate instanceof True){
							node.setScore(null);

							if(defaultChild != null){
								node.setDefaultChild(null);
							} // End if

							if(miningFunction == MiningFunction.REGRESSION){
								initScore(node, child);
							} else

							if(miningFunction == MiningFunction.CLASSIFICATION){
								initScoreDistribution(node, child);
							}

							initDefaultChild(node, child);
							replaceChildWithGrandchildren(node, child);

							return;
						}
					}
				}

				super.exitNode(node);
			}

			@Override
			public void enterTreeModel(TreeModel treeModel){
				super.enterTreeModel(treeModel);

				this.miningFunction = treeModel.requireMiningFunction();
			}

			@Override
			public void exitTreeModel(TreeModel treeModel){
				super.exitTreeModel(treeModel);

				this.miningFunction = null;
			}
		};
		nodePruner.applyTo(model);

		Visitor nullSegmentRemover = new AbstractVisitor(){

			@Override
			public VisitorAction visit(Segmentation segmentation){
				List<Segment> segments = segmentation.getSegments();

				for(Iterator<Segment> it = segments.iterator(); it.hasNext(); ){
					Segment segment = it.next();

					Model model = segment.requireModel();
					if(model instanceof TreeModel){
						TreeModel treeModel = (TreeModel)model;

						if(isNull(treeModel)){
							it.remove();
						}
					}
				}

				return super.visit(segmentation);
			}

			private boolean isNull(TreeModel treeModel){
				Node root = treeModel.requireNode();

				if(!root.hasNodes()){
					Predicate predicate = root.requirePredicate();

					if(predicate instanceof True){
						Number score = (Number)root.getScore();

						return score.doubleValue() == 0d;
					}
				}

				return false;
			}
		};
		nullSegmentRemover.applyTo(model);

		return model;
	}

	protected boolean hasTreeOptions(){
		E model = getModel();

		return (model instanceof HasTreeOptions);
	}

	public E getModel(){
		return getModel("model");
	}

	public Map<String, E> getModels(){
		return getModels("models");
	}
}