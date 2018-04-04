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
import java.util.List;

import com.google.common.io.ByteStreams;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PickleUtilTest {

	@Test
	public void python27() throws IOException {
		unpickle("python-2.7_joblib-0.9.4.pkl.z");

		unpickle("python-2.7_pickle-p2.pkl");

		unpickle("python-2.7_sklearn-joblib-0.9.4.pkl.z");
		unpickle("python-2.7_sklearn-joblib-0.10.2.pkl.z");

		unpickleNumpyArrays("python-2.7_numpy-1.11.2");
	}

	@Test
	public void python34() throws IOException {
		unpickle("python-3.4_joblib-0.9.3.pkl.z");
		unpickle("python-3.4_joblib-0.9.4.pkl.z");
		unpickle("python-3.4_joblib-0.10.0.pkl.z");

		unpickle("python-3.4_pickle-p2.pkl");
		unpickle("python-3.4_pickle-p3.pkl");
		unpickle("python-3.4_pickle-p4.pkl");

		unpickle("python-3.4_sklearn-joblib-0.9.4.pkl.z");
		unpickle("python-3.4_sklearn-joblib-0.11.pkl.z");

		unpickleNumpyArrays("python-3.4_numpy-1.13.3");
	}

	private void unpickleNumpyArrays(String prefix) throws IOException {
		unpickleNumpyArray(prefix + "_int8.pkl", Byte.MIN_VALUE, Byte.MAX_VALUE, 1);
		unpickleNumpyArray(prefix + "_int16.pkl", Short.MIN_VALUE, Short.MAX_VALUE, 127);

		String[] dtypes = {"int32", "int64", "float32", "float64"};
		for(String dtype : dtypes){
			unpickleNumpyArray(prefix + "_" + dtype + ".pkl", Integer.MIN_VALUE, Integer.MAX_VALUE, 64 * 32767);
		}
	}

	private void unpickleNumpyArray(String name, int min, int max, int step) throws IOException {
		HasArray hasArray = (HasArray)unpickle(name);

		List<?> values = hasArray.getArrayContent();

		for(int i = 0; i < values.size(); i++){
			Number expectedValue = min + (i * step);
			Number value = (Number)values.get(i);

			if(value instanceof Float){
				assertEquals((Float)expectedValue.floatValue(), (Float)value);
			} else

			if(value instanceof Double){
				assertEquals((Double)expectedValue.doubleValue(), (Double)value);
			} else

			{
				assertEquals(expectedValue.intValue(), value.intValue());
			}
		}
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