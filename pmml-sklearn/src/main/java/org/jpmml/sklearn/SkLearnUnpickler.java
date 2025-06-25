/*
 * Copyright (c) 2025 Villu Ruusmann
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

import java.io.IOException;
import java.util.Objects;

import net.razorvine.pickle.Opcodes;
import net.razorvine.pickle.objects.ClassDict;
import org.jpmml.python.JoblibUnpickler;
import sklearn2pmml.SkLearn2PMMLFields;

public class SkLearnUnpickler extends JoblibUnpickler {

	@Override
	protected Object dispatch(short key) throws IOException {
		Object result = super.dispatch(key);

		if(key == Opcodes.BUILD){
			Object head = peekHead();

			if(Objects.equals(ClassDict.class, head.getClass())){
				ClassDict dict = (ClassDict)head;

				if(dict.containsKey(SkLearn2PMMLFields.PMML_BASE_CLASS)){
					ExtendedClassDict object = ExtendedClassDict.build(dict.getClassName());
					object.__setstate__(dict);

					replaceHead(object);
				}
			}
		}

		return result;
	}
}