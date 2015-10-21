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
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import com.google.common.collect.Iterators;
import com.google.common.collect.Ordering;
import com.google.common.primitives.Ints;

public class ZipFileStorage implements Storage {

	private ZipFile zipFile = null;


	public ZipFileStorage(ZipFile zipFile){
		this.zipFile = zipFile;
	}

	@Override
	public InputStream getObject() throws IOException {
		Iterator<? extends ZipEntry> entries = Iterators.forEnumeration(this.zipFile.entries());

		Ordering<ZipEntry> ordering = new Ordering<ZipEntry>(){

			@Override
			public int compare(ZipEntry left, ZipEntry right){
				String leftName = left.getName();
				String rightName = right.getName();

				return Ints.compare(leftName.length(), rightName.length());
			}
		};

		ZipEntry entry = ordering.min(entries);

		return getInputStream(entry);
	}

	@Override
	public InputStream getArray(String path) throws IOException {
		ZipEntry entry = this.zipFile.getEntry(path);

		return getInputStream(entry);
	}

	private InputStream getInputStream(ZipEntry entry) throws IOException {
		return this.zipFile.getInputStream(entry);
	}

	@Override
	public void close() throws IOException {
		this.zipFile.close();
	}

	static
	public ZipFileStorage open(File file){

		try {
			ZipFile zipFile = new ZipFile(file);

			return new ZipFileStorage(zipFile);
		} catch(IOException ioe){
			return null;
		}
	}
}