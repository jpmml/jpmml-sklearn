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
package org.jpmml.sklearn;

import org.junit.Test;
import sklearn.tree.Tree;

import static org.junit.Assert.assertEquals;

public class ClassDictUtilTest {

	@Test
	public void formatMember(){
		ExtensionObjectConstructor constructor = new ExtensionObjectConstructor("sklearn.tree._tree", "Tree", Tree.class);

		Tree tree = (Tree)constructor.construct(new Object[]{0, 0, 0});

		assertEquals("sklearn.tree._tree.Tree.values", ClassDictUtil.formatMember(tree, "values"));
	}
}