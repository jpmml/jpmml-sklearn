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
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import causalml.meta.visitors.TreeModelGroupActivator;
import com.google.common.collect.Iterables;
import org.dmg.pmml.Model;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
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
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;
import org.jpmml.converter.visitors.TreeModelPruner;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.python.ClassDictUtil;
import sklearn.Estimator;

abstract
public class BaseSLearner<E extends Estimator> extends BaseLearner<E> {

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

			Schema binarySchema = schema.toRelabeledSchema(continuousLabel);

			BinaryFeature controlFeature = new BinaryFeature(encoder, groupFeature, controlName);

			// XXX
			List<Feature> segmentFeatures = (List<Feature>)binarySchema.getFeatures();
			segmentFeatures.set(0, controlFeature);

			Model binaryModel = encodeBinaryModel(estimator, groupName, controlName, binarySchema);

			binaryModels.add(binaryModel);
		}

		if(binaryModels.size() == 1){
			return Iterables.getOnlyElement(binaryModels);
		} else

		{
			return MiningModelUtil.createMultiModelChain(binaryModels, Segmentation.MissingPredictionTreatment.RETURN_MISSING);
		}
	}

	protected Model encodeBinaryModel(E estimator, String groupName, String controlName, Schema schema){
		Model treatmentModel = encodeEstimator(estimator, schema);
		Model controlModel = encodeEstimator(estimator, schema);

		Visitor nullBranchMarker = new AbstractTreeModelTransformer(){

			private Set<Node> vitalNodes = new HashSet<>();


			@Override
			public void enterNode(Node node){
				Number recordCount = node.getRecordCount();

				if(recordCount != null){
					node.setRecordCount(null);
				}
			}

			@Override
			public void exitNode(Node node){
				Predicate predicate = node.requirePredicate();

				if(hasFieldReference(predicate, groupName)){
					this.vitalNodes.add(node);
				} // End if

				if(this.vitalNodes.contains(node)){
					Node parentNode = getParentNode();

					if(parentNode != null){
						this.vitalNodes.add(parentNode);
					}

					return;
				}

				Node treatmentAncestor = getAncestorNode(parentNode -> hasFieldReference(parentNode.requirePredicate(), groupName));
				if(treatmentAncestor == null){
					node.setScore(0d);
				}
			}

			@Override
			public void enterTreeModel(TreeModel treeModel){
				super.enterTreeModel(treeModel);
			}

			@Override
			public void exitTreeModel(TreeModel treeModel){
				super.exitTreeModel(treeModel);

				this.vitalNodes.clear();
			}
		};
		nullBranchMarker.applyTo(controlModel);
		nullBranchMarker.applyTo(treatmentModel);

		Visitor branchPruner = new AbstractTreeModelTransformer(){

			@Override
			public void exitNode(Node node){
				Object defaultChild = node.getDefaultChild();
				Object score = node.getScore();

				if(node.hasNodes()){
					List<Node> children = node.getNodes();

					Object childScore = getConstantScore(children);
					if(Objects.equals(score, childScore)){

						if(defaultChild != null){
							node.setDefaultChild(null);
						}

						children.clear();
					}
				}
			}

			private Object getConstantScore(List<Node> nodes){
				Object result = null;

				for(Node node : nodes){
					Object score = node.getScore();

					if(node.hasNodes()){
						return null;
					} // End if

					if(score == null){
						return null;
					} // End if

					if(result == null){
						result = score;
					} else

					{
						if(!Objects.equals(score, result)){
							return null;
						}
					}
				}

				return result;
			}
		};
		branchPruner.applyTo(treatmentModel);
		branchPruner.applyTo(controlModel);

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
		treatmentActivator.applyTo(treatmentModel);

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
		controlActivator.applyTo(controlModel);

		Visitor nodePruner = new TreeModelPruner(){

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
							}

							initScore(node, child);
							initDefaultChild(node, child);
							replaceChildWithGrandchildren(node, child);

							return;
						}
					}
				}

				super.exitNode(node);
			}
		};
		nodePruner.applyTo(treatmentModel);
		nodePruner.applyTo(controlModel);

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
		nullSegmentRemover.applyTo(treatmentModel);
		nullSegmentRemover.applyTo(controlModel);

		return encodeBinaryModel(treatmentModel, controlModel, schema);
	}

	public Map<String, E> getModels(){
		return getModels("models");
	}
}