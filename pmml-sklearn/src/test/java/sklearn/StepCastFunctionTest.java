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
package sklearn;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.jpmml.python.ClassDictConstructorUtil;
import org.jpmml.python.PythonObjectConstructor;
import org.jpmml.sklearn.SkLearnUtil;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class StepCastFunctionTest {

	@Test
	public void getInstantiableSubclassConstructors(){
		assertEquals(createClassNames("sklearn2pmml.decoration", "CategoricalDomain", "ContinuousDomain", "DateDomain", "DateTimeDomain", "OrdinalDomain"), getInstantiableSubclassNames("sklearn2pmml.decoration.Domain"));
		assertEquals(createClassNames("sklearn2pmml.decoration", "CategoricalDomain", "OrdinalDomain"), getInstantiableSubclassNames("sklearn2pmml.decoration.DiscreteDomain"));
		assertEquals(createClassNames("sklearn2pmml.decoration", "DateDomain", "DateTimeDomain"), getInstantiableSubclassNames("sklearn2pmml.decoration.TemporalDomain"));
		assertEquals(createClassNames("sklearn2pmml.decoration", "ContinuousDomainEraser", "DiscreteDomainEraser"), getInstantiableSubclassNames("sklearn2pmml.decoration.DomainEraser"));
		assertEquals(createClassNames("sklearn2pmml.decoration", "Alias", "MultiAlias"), getInstantiableSubclassNames("sklearn2pmml.decoration.TransformerWrapper"));

		assertEquals(createClassNames("sklearn2pmml.preprocessing", "DaysSinceYearTransformer", "SecondsSinceYearTransformer"), getInstantiableSubclassNames("sklearn2pmml.preprocessing.DurationTransformer"));
		assertEquals(createClassNames("sklearn2pmml.preprocessing", "MatchesTransformer", "ReplaceTransformer"), getInstantiableSubclassNames("sklearn2pmml.preprocessing.RegExTransformer"));
		assertEquals(createClassNames("sklearn2pmml.preprocessing", "StringLengthTransformer", "StringNormalizer", "SubstringTransformer", "WordCountTransformer"), getInstantiableSubclassNames("sklearn2pmml.preprocessing.StringTransformer"));

		assertEquals(createClassNames("sklearn2pmml.util", "Expression", "Predicate"), getInstantiableSubclassNames("sklearn2pmml.util.Evaluatable"));
	}

	static
	private List<String> getInstantiableSubclassNames(String className){
		List<PythonObjectConstructor> pythonObjectConstructors = StepCastFunction.getInstantiableSubclassConstructors(className);

		return pythonObjectConstructors.stream()
			.map(ClassDictConstructorUtil::getClassName)
			.sorted()
			.collect(Collectors.toList());
	}

	static
	private List<String> createClassNames(String module, String... names){
		return Arrays.stream(names)
			.map(name -> module + "." + name)
			.collect(Collectors.toList());
	}

	static {
		SkLearnUtil.initOnce();
	}
}
