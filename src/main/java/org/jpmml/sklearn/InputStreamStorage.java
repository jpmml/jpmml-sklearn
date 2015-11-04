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

import java.io.IOException;
import java.io.InputStream;

public class InputStreamStorage implements Storage {

	private InputStream is = null;


	public InputStreamStorage(InputStream is){
		this.is = is;
	}

	@Override
	public InputStream getObject() throws IOException {
		return ensureOpen();
	}

	@Override
	public InputStream getArray(String name){
		throw new UnsupportedOperationException();
	}

	@Override
	public void close() throws IOException {

		try {
			if(this.is != null){
				this.is.close();
			}
		} finally {
			this.is = null;
		}
	}

	private InputStream ensureOpen() throws IOException {

		if(this.is == null){
			throw new IOException();
		}

		return this.is;
	}
}