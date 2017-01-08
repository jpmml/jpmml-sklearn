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

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;
import java.util.zip.InflaterInputStream;

import com.google.common.io.ByteStreams;
import com.google.common.io.CountingInputStream;

public class CompressedInputStreamStorage extends InputStreamStorage {

	public CompressedInputStreamStorage(InputStream is) throws IOException {
		super(init(new PushbackInputStream(is, 2)));
	}

	static
	private InputStream init(PushbackInputStream is) throws IOException {
		byte[] magic = new byte[2];

		ByteStreams.readFully(is, magic);

		is.unread(magic);

		// Joblib 0.10.0 and newer
		if(magic[0] == 'x'){
			return initZlib(is);
		} else

		// Joblib 0.9.4 and earlier
		if(magic[0] == 'Z' && magic[1] == 'F'){
			return initCompat(is);
		} else

		{
			throw new IOException();
		}
	}

	static
	private InputStream initZlib(PushbackInputStream is){
		InflaterInputStream zlibIs = new InflaterInputStream(is);

		return zlibIs;
	}

	static
	private InputStream initCompat(PushbackInputStream is) throws IOException {
		byte[] headerBytes = new byte[2 + 19];

		ByteStreams.readFully(is, headerBytes);

		String header = new String(headerBytes);

		if(!header.startsWith("ZF0x")){
			throw new IOException();
		}

		// Remove trailing whitespace
		header = header.trim();

		final
		long expectedSize = Long.parseLong(header.substring(4), 16);

		// Consume the first byte
		int firstByte = is.read();
		if(firstByte < 0){
			return is;
		} // End if

		// If the first byte is not a space character, then make it available for reading again
		if(firstByte != '\u0020'){
			is.unread(firstByte);
		}

		InflaterInputStream zlibIs = new InflaterInputStream(is);

		InputStream result = new FilterInputStream(new CountingInputStream(zlibIs)){

			private boolean closed = false;


			@Override
			public void close() throws IOException {

				if(this.closed){
					return;
				}

				this.closed = true;

				long size = ((CountingInputStream)super.in).getCount();

				super.close();

				if(size != expectedSize){
					throw new IOException("Expected " + expectedSize + " byte(s) of uncompressed data, got " + size + " byte(s)");
				}
			}
		};

		return result;
	}
}
