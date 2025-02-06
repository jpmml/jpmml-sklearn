/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sklearn2pmml.preprocessing;

import org.dmg.pmml.Apply;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MatchesTransformerTest extends RegExTransformerTest {

	@Test
	public void matches(){
		MatchesTransformer matchesTransformer = new MatchesTransformer()
			.setPattern("ar?y")
			.setReFlavour(RegExTransformer.RE_FLAVOUR_RE);

		Apply apply = encode(matchesTransformer);

		assertTrue((Boolean)evaluate(apply, "January"));
		assertFalse((Boolean)evaluate(apply, "March"));
		assertTrue((Boolean)evaluate(apply, "May"));

		matchesTransformer = matchesTransformer
			.setPattern("r$");

		apply = encode(matchesTransformer);

		assertFalse((Boolean)evaluate(apply, "March"));
		assertTrue((Boolean)evaluate(apply, "October"));
	}
}