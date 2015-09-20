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
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import com.google.common.collect.Ordering;
import com.google.common.io.ByteStreams;
import com.google.common.primitives.Ints;

public class ZipStreamStorage implements Storage {

	private Map<String, byte[]> entries = new LinkedHashMap<>();


	public ZipStreamStorage(InputStream is) throws IOException {
		ZipInputStream zis = new ZipInputStream(is);

		while(true){
			ZipEntry entry = zis.getNextEntry();
			if(entry == null){
				break;
			}

			this.entries.put(entry.getName(), ByteStreams.toByteArray(zis));

			zis.closeEntry();
		}
	}

	@Override
	public InputStream getObject(){
		Set<String> names = this.entries.keySet();

		Ordering<String> ordering = new Ordering<String>(){

			@Override
			public int compare(String left, String right){
				return Ints.compare(left.length(), right.length());
			}
		};

		String name = ordering.min(names);

		return getInputStream(name);
	}

	@Override
	public InputStream getArray(String name){
		return getInputStream(name);
	}

	private InputStream getInputStream(String name){
		byte[] buffer = this.entries.get(name);

		if(buffer == null){
			throw new NullPointerException();
		}

		return new ByteArrayInputStream(buffer);
	}

	@Override
	public void close(){
		this.entries.clear();
	}
}