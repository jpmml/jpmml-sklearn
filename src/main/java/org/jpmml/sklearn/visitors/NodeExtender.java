/*
 * Copyright (c) 2018 Villu Ruusmann
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

import java.util.List;

import org.dmg.pmml.Extension;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.tree.Node;
import org.jpmml.model.visitors.AbstractVisitor;

abstract
public class NodeExtender extends AbstractVisitor {

	private String name = null;


	public NodeExtender(String name){
		setName(name);
	}

	abstract
	public String getValue(Node node);

	@Override
	public VisitorAction visit(Node node){
		String name = getName();
		String value = getValue(node);

		if(value != null){
			List<Extension> extensions = node.getExtensions();

			Extension extension = new Extension()
				.setName(name)
				.setValue(value);

			extensions.add(extension);
		}

		return super.visit(node);
	}

	public String getName(){
		return this.name;
	}

	private void setName(String name){
		this.name = name;
	}
}