/*
 * Copyright (c) 2017 Villu Ruusmann
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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class VersionUtilTest {

	@Test
	public void compareVersion(){
		assertTrue(VersionUtil.compareVersion("0.18", "0.19") < 0);

		assertTrue(VersionUtil.compareVersion("0.19", "0.19") == 0);
		assertTrue(VersionUtil.compareVersion("0.19.0", "0.19") == 0);
		assertTrue(VersionUtil.compareVersion("0.19.1", "0.19") == 0);

		assertTrue(VersionUtil.compareVersion("0.19", "0.19.0") == 0);
		assertTrue(VersionUtil.compareVersion("0.19.0", "0.19.0") == 0);
		assertTrue(VersionUtil.compareVersion("0.19.1", "0.19.0") > 0);

		assertTrue(VersionUtil.compareVersion("0.19", "0.19.1") < 0);
		assertTrue(VersionUtil.compareVersion("0.19.0", "0.19.1") < 0);
		assertTrue(VersionUtil.compareVersion("0.19.1", "0.19.1") == 0);

		assertTrue(VersionUtil.compareVersion("0.20", "0.19") > 0);

		assertTrue(VersionUtil.compareVersion("0.22.2", "0.22.2.post1") == 0);
	}

	@Test
	public void parseVersion(){
		assertEquals(Arrays.asList(0, 19), VersionUtil.parseVersion("0.19"));
		assertEquals(Arrays.asList(0, 19), VersionUtil.parseVersion("0.19a1"));
		assertEquals(Arrays.asList(0, 19), VersionUtil.parseVersion("0.19b1"));
		assertEquals(Arrays.asList(0, 19), VersionUtil.parseVersion("0.19rc1"));
		assertEquals(Arrays.asList(0, 19), VersionUtil.parseVersion("0.19.dev"));
		assertEquals(Arrays.asList(0, 19), VersionUtil.parseVersion("0.19.dev1"));
		assertEquals(Arrays.asList(0, 19, 1), VersionUtil.parseVersion("0.19.1"));

		assertEquals(Arrays.asList(0, 22), VersionUtil.parseVersion("0.22.post1"));
		assertEquals(Arrays.asList(0, 22), VersionUtil.parseVersion("0.22rc1.post1.dev1"));
		assertEquals(Arrays.asList(0, 22, 2), VersionUtil.parseVersion("0.22.2"));
		assertEquals(Arrays.asList(0, 22, 2), VersionUtil.parseVersion("0.22.2.post1"));
		assertEquals(Arrays.asList(0, 22, 2), VersionUtil.parseVersion("0.22.2.post1.dev1"));
	}
}