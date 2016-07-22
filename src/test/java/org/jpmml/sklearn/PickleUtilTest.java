/*
 * Copyright (c) 2016 Villu Ruusmann
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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

import com.google.common.io.ByteStreams;
import org.junit.Test;

public class PickleUtilTest {

	@Test
	public void python27() throws IOException {
		unpickle("python-2.7-joblib-0.9.4.pkl.z");

		unpickle("python-2.7-pickle-p2.pkl");

		unpickle("python-2.7-sklearn_joblib-0.9.4.pkl.z");
	}

	@Test
	public void python34() throws IOException {
		unpickle("python-3.4-joblib-0.9.3.pkl.z");
		unpickle("python-3.4-joblib-0.9.4.pkl.z");
		unpickle("python-3.4-joblib-0.10.0.pkl.z");

		unpickle("python-3.4-pickle-p2.pkl");
		unpickle("python-3.4-pickle-p3.pkl");
		unpickle("python-3.4-pickle-p4.pkl");

		unpickle("python-3.4-sklearn_joblib-0.9.4.pkl.z");
	}

	static
	private Object unpickle(String name) throws IOException {
		byte[] bytes;

		InputStream is = PickleUtilTest.class.getResourceAsStream("/dump/" + name);

		try {
			bytes = ByteStreams.toByteArray(is);
		} finally {
			is.close();
		}

		return unpickle(bytes);
	}

	static
	private Object unpickle(byte[] bytes) throws IOException {
		Storage storage;

		try {
			storage = new CompressedInputStreamStorage(new ByteArrayInputStream(bytes));
		} catch(IOException ioe){
			storage = new InputStreamStorage(new ByteArrayInputStream(bytes));
		}

		return PickleUtil.unpickle(storage);
	}
}