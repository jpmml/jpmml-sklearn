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
package causalml.meta.visitors;

import java.util.List;

import org.dmg.pmml.False;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.Node;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;
import org.jpmml.model.UnsupportedElementException;

abstract
public class TreeModelGroupActivator extends AbstractTreeModelTransformer {

	abstract
	public Boolean getActivation(Predicate predicate);

	@Override
	public void enterNode(Node node){
	}

	@Override
	public void exitNode(Node node){
		Predicate predicate = node.requirePredicate();

		Boolean activation = getActivation(predicate);
		if(activation != null){
			node.setPredicate(activation ? True.INSTANCE : False.INSTANCE);
		} // End if

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			if(children.size() != 2){
				throw new UnsupportedElementException(node);
			}

			Node firstChild = children.get(0);
			Node secondChild = children.get(1);

			Predicate firstPredicate = firstChild.requirePredicate();
			Predicate secondPredicate = secondChild.requirePredicate();

			if((firstPredicate instanceof False) && (secondPredicate instanceof True)){
				children = swapChildren(node);
			}
		}
	}
}