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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

import com.google.common.io.ByteStreams;

public class InputStreamStorage implements Storage {

	private byte[] buffer = null;


	public InputStreamStorage(InputStream is) throws IOException {
		this.buffer = ByteStreams.toByteArray(is);
	}

	@Override
	public InputStream getObject() throws IOException {

		if(this.buffer == null){
			throw new IOException();
		}

		return new ByteArrayInputStream(this.buffer);
	}

	@Override
	public InputStream getArray(String name){
		throw new UnsupportedOperationException();
	}

	@Override
	public void close(){
		this.buffer = null;
	}
}