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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class FileStorage implements Storage {

	private File file = null;


	public FileStorage(File file){
		this.file = file;
	}

	@Override
	public InputStream getObject() throws IOException {
		return new FileInputStream(ensureOpen());
	}

	@Override
	public InputStream getArray(String name) throws IOException {
		File file = ensureOpen();

		File arrayFile = new File(file.getParentFile(), name);

		return new FileInputStream(arrayFile);
	}

	@Override
	public void close(){
		this.file = null;
	}

	private File ensureOpen() throws IOException {

		if(this.file == null){
			throw new IOException();
		}

		return this.file;
	}
}