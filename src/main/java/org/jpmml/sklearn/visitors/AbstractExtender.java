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
import org.dmg.pmml.HasExtensions;
import org.dmg.pmml.PMMLObject;
import org.jpmml.model.visitors.AbstractVisitor;

abstract
public class AbstractExtender extends AbstractVisitor {

	private String name = null;


	public AbstractExtender(String name){
		setName(name);
	}

	public <E extends PMMLObject & HasExtensions<E>> void addExtension(E object, String value){
		String name = getName();

		Extension extension = new Extension()
			.setName(name)
			.setValue(value);

		List<Extension> extensions = object.getExtensions();

		extensions.add(extension);
	}

	public String getName(){
		return this.name;
	}

	private void setName(String name){
		this.name = name;
	}
}