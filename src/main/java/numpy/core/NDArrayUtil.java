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
package numpy.core;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.io.ByteStreams;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.common.primitives.UnsignedInts;
import net.razorvine.pickle.Unpickler;
import net.razorvine.serpent.Parser;
import net.razorvine.serpent.ast.Ast;
import numpy.DType;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sklearn.TupleUtil;

public class NDArrayUtil {

	private NDArrayUtil(){
	}

	static
	public int[] getShape(NDArray array){
		Object[] shape = array.getShape();

		List<? extends Number> values = (List)Arrays.asList(shape);

		return Ints.toArray(ValueUtil.asIntegers(values));
	}

	/**
	 * Gets the payload of a one-dimensional array.
	 */
	static
	public List<?> getContent(NDArray array){
		Object content = array.getContent();

		return asJavaList(array, (List<?>)content);
	}

	/**
	 * Gets the payload of the specified dimension of a multi-dimensional array.
	 *
	 * @param key The dimension.
	 */
	static
	public List<?> getContent(NDArray array, String key){
		Map<String, ?> content = (Map<String, ?>)array.getContent();

		return asJavaList(array, (List<?>)content.get(key));
	}

	static
	private <E> List<E> asJavaList(NDArray array, List<E> values){
		boolean fortranOrder = array.getFortranOrder();

		if(fortranOrder){
			int[] shape = getShape(array);

			switch(shape.length){
				case 1:
					return values;
				case 2:
					return toJavaList(values, shape[0], shape[1]);
				default:
					throw new IllegalArgumentException();
			}
		}

		return values;
	}

	/**
	 * Translates a column-major (ie. Fortran-type) array to a row-major (ie. C-type) array.
	 */
	static
	private <E> List<E> toJavaList(List<E> values, int rows, int columns){
		List<E> result = new ArrayList<>(values.size());

		for(int i = 0; i < values.size(); i++){
			int row = i / columns;
			int column = i % columns;

			E value = values.get((column * rows) + row);

			result.add(value);
		}

		return result;
	}

	/**
	 * http://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
	 */
	static
	public NDArray parseNpy(InputStream is) throws IOException {
		byte[] magicBytes = new byte[MAGIC_STRING.length];

		ByteStreams.readFully(is, magicBytes);

		if(!Arrays.equals(magicBytes, MAGIC_STRING)){
			throw new IOException();
		}

		int majorVersion = readUnsignedByte(is);
		int minorVersion = readUnsignedByte(is);

		if(majorVersion != 1 || minorVersion != 0){
			throw new IOException();
		}

		int headerLength = readUnsignedShort(is, ByteOrder.LITTLE_ENDIAN);

		if(headerLength < 0){
			throw new IOException();
		}

		byte[] headerBytes = new byte[headerLength];

		ByteStreams.readFully(is, headerBytes);

		String header = new String(headerBytes);

		// Remove trailing whitespace
		header = header.trim();

		Map<String, ?> headerDict = parseDict(header);

		Object descr = headerDict.get("descr");
		Boolean fortranOrder = (Boolean)headerDict.get("fortran_order");
		Object[] shape = (Object[])headerDict.get("shape");

		byte[] data = ByteStreams.toByteArray(is);

		NDArray array = new NDArray();
		array.__setstate__(new Object[]{Arrays.asList(majorVersion, minorVersion), shape, descr, fortranOrder, data});

		return array;
	}

	static
	public Object parseData(InputStream is, Object descr, Object[] shape) throws IOException {

		if(descr instanceof DType){
			DType dType = (DType)descr;

			descr = dType.toDescr();
		}

		int length = 1;

		for(int i = 0; i < shape.length; i++){
			length *= ValueUtil.asInt((Number)shape[i]);
		} // End if

		if(descr instanceof String){
			return parseArray(is, (String)descr, length);
		}

		List<Object[]> dims = (List<Object[]>)descr;

		Map<String, List<?>> result = new LinkedHashMap<>();

		List<Object[]> objects = parseMultiArray(is, (List)TupleUtil.extractElementList(dims, 1), length);

		for(int i = 0; i < dims.size(); i++){
			Object[] dim = dims.get(i);

			result.put((String)dim[0], TupleUtil.extractElementList(objects, i));
		}

		return result;
	}

	static
	public List<Object> parseArray(InputStream is, String descr, int length) throws IOException {
		List<Object> result = new ArrayList<>(length);

		TypeDescriptor descriptor = new TypeDescriptor(descr);

		while(result.size() < length){
			Object element = descriptor.read(is);

			if(descriptor.isObject()){
				NDArray array = (NDArray)element;

				result.addAll(NDArrayUtil.getContent(array));

				continue;
			}

			result.add(element);
		}

		return result;
	}

	static
	public List<Object[]> parseMultiArray(InputStream is, List<String> descrs, int length) throws IOException {
		List<Object[]> result = new ArrayList<>(length);

		List<TypeDescriptor> descriptors = new ArrayList<>();

		for(String descr : descrs){
			TypeDescriptor descriptor = new TypeDescriptor(descr);

			if(descriptor.isObject()){
				throw new IllegalArgumentException(descr);
			}

			descriptors.add(descriptor);
		}

		for(int i = 0; i < length; i++){
			Object[] element = new Object[descriptors.size()];

			for(int j = 0; j < descriptors.size(); j++){
				TypeDescriptor descriptor = descriptors.get(j);

				element[j] = descriptor.read(is);
			}

			result.add(element);
		}

		return result;
	}

	static
	private Map<String, ?> parseDict(String string){
		Parser parser = new Parser();

		Ast ast = parser.parse(string);

		return (Map<String, ?>)ast.getData();
	}

	static
	private byte readByte(InputStream is) throws IOException {
		int b = is.read();
		if(b < 0){
			throw new EOFException();
		}

		return (byte)b;
	}

	static
	private int readUnsignedByte(InputStream is) throws IOException {
		int b = is.read();
		if(b < 0){
			throw new EOFException();
		}

		return b;
	}

	static
	private int readUnsignedShort(InputStream is, ByteOrder byteOrder) throws IOException {
		byte b1 = readByte(is);
		byte b2 = readByte(is);

		if((ByteOrder.BIG_ENDIAN).equals(byteOrder)){
			return Ints.fromBytes((byte)0, (byte)0, b1, b2);
		} else

		if((ByteOrder.LITTLE_ENDIAN).equals(byteOrder)){
			return Ints.fromBytes((byte)0, (byte)0, b2, b1);
		}

		throw new IOException();
	}

	static
	private int readInt(InputStream is, ByteOrder byteOrder) throws IOException {
		byte b1 = readByte(is);
		byte b2 = readByte(is);
		byte b3 = readByte(is);
		byte b4 = readByte(is);

		if((ByteOrder.BIG_ENDIAN).equals(byteOrder)){
			return Ints.fromBytes(b1, b2, b3, b4);
		} else

		if((ByteOrder.LITTLE_ENDIAN).equals(byteOrder)){
			return Ints.fromBytes(b4, b3, b2, b1);
		}

		throw new IOException();
	}

	static
	private long readLong(InputStream is, ByteOrder byteOrder) throws IOException {
		byte b1 = readByte(is);
		byte b2 = readByte(is);
		byte b3 = readByte(is);
		byte b4 = readByte(is);
		byte b5 = readByte(is);
		byte b6 = readByte(is);
		byte b7 = readByte(is);
		byte b8 = readByte(is);

		if((ByteOrder.BIG_ENDIAN).equals(byteOrder)){
			return Longs.fromBytes(b1, b2, b3, b4, b5, b6, b7, b8);
		} else

		if((ByteOrder.LITTLE_ENDIAN).equals(byteOrder)){
			return Longs.fromBytes(b8, b7, b6, b5, b4, b3, b2, b1);
		}

		throw new IOException();
	}

	static
	private float readFloat(InputStream is, ByteOrder byteOrder) throws IOException {
		return Float.intBitsToFloat(readInt(is, byteOrder));
	}

	static
	private double readDouble(InputStream is, ByteOrder byteOrder) throws IOException {
		return Double.longBitsToDouble(readLong(is, byteOrder));
	}

	static
	private Object readObject(InputStream is) throws IOException {
		Unpickler unpickler = new Unpickler();

		return unpickler.load(is);
	}

	static
	private String readString(InputStream is, int size) throws IOException {
		byte[] buffer = new byte[size];

		ByteStreams.readFully(is, buffer);

		return toString(buffer, "UTF-8");
	}

	static
	private String readUnicode(InputStream is, ByteOrder byteOrder, int size) throws IOException {
		byte[] buffer = new byte[size * 4];

		ByteStreams.readFully(is, buffer);

		if((ByteOrder.BIG_ENDIAN).equals(byteOrder)){
			return toString(buffer, "UTF-32BE");
		} else

		if((ByteOrder.LITTLE_ENDIAN).equals(byteOrder)){
			return toString(buffer, "UTF-32LE");
		}

		throw new IOException();
	}

	static
	private String toString(byte[] buffer, String encoding) throws IOException {
		String string = new String(buffer, encoding);

		// Trim trailing zero characters
		while(string.length() > 0 && string.charAt(string.length() - 1) == '\0'){
			string = string.substring(0, string.length() - 1);
		}

		return string;
	}

	/**
	 * http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
	 * http://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.byteorder.html
	 */
	static
	private class TypeDescriptor {

		private ByteOrder byteOrder = null;

		private Kind kind = null;

		private int size = 0;


		private TypeDescriptor(String descr){
			int i = 0;

			ByteOrder byteOrder = null;

			switch(descr.charAt(i)){
				// Native
				case '=':
					byteOrder = ByteOrder.nativeOrder();
					i++;
					break;
				// Big-endian
				case '>':
					byteOrder = ByteOrder.BIG_ENDIAN;
					i++;
					break;
				// Little-endian
				case '<':
					byteOrder = ByteOrder.LITTLE_ENDIAN;
					i++;
					break;
				// Not applicable
				case '|':
					i++;
					break;
			}

			setByteOrder(byteOrder);

			Kind kind = Kind.forChar(descr.charAt(i));

			i++;

			setKind(kind);

			if(i < descr.length()){
				int size = Integer.parseInt(descr.substring(i));

				setSize(size);
			}
		}

		public Object read(InputStream is) throws IOException {
			Kind kind = getKind();
			ByteOrder byteOrder = getByteOrder();
			int size = getSize();

			switch(kind){
				case BOOLEAN:
					{
						switch(size){
							case 1:
								return (readByte(is) == 1);
							default:
								break;
						}
					}
					break;
				case INTEGER:
					{
						switch(size){
							case 4:
								return readInt(is, byteOrder);
							case 8:
								return readLong(is, byteOrder);
							default:
								break;
						}
					}
					break;
				case UNSIGNED_INTEGER:
					{
						switch(size){
							case 4:
								return UnsignedInts.toLong(readInt(is, byteOrder));
							default:
								break;
						}
					}
					break;
				case FLOAT:
					{
						switch(size){
							case 4:
								return readFloat(is, byteOrder);
							case 8:
								return readDouble(is, byteOrder);
							default:
								break;
						}
					}
					break;
				case OBJECT:
					{
						return readObject(is);
					}
				case STRING:
					{
						return readString(is, size);
					}
				case UNICODE:
					{
						return readUnicode(is, byteOrder, size);
					}
				case VOID:
					{
						byte[] buffer = new byte[size];

						ByteStreams.readFully(is, buffer);

						return buffer;
					}
				default:
					break;
			}

			throw new IOException();
		}

		public boolean isObject(){
			Kind kind = getKind();

			switch(kind){
				case OBJECT:
					return true;
				default:
					return false;
			}
		}

		public ByteOrder getByteOrder(){
			return this.byteOrder;
		}

		private void setByteOrder(ByteOrder byteOrder){
			this.byteOrder = byteOrder;
		}

		public Kind getKind(){
			return this.kind;
		}

		private void setKind(Kind kind){
			this.kind = kind;
		}

		public int getSize(){
			return this.size;
		}

		private void setSize(int size){
			this.size = size;
		}

		static
		private enum Kind {
			BOOLEAN,
			INTEGER,
			UNSIGNED_INTEGER,
			FLOAT,
			COMPLEX_FLOAT,
			OBJECT,
			STRING,
			UNICODE,
			VOID,
			;

			static
			public Kind forChar(char c){

				switch(c){
					case 'b':
						return BOOLEAN;
					case 'i':
						return INTEGER;
					case 'u':
						return UNSIGNED_INTEGER;
					case 'f':
						return FLOAT;
					case 'c':
						return COMPLEX_FLOAT;
					case 'O':
						return OBJECT;
					case 'S':
					case 'a':
						return STRING;
					case 'U':
						return UNICODE;
					case 'V':
						return VOID;
					default:
						throw new IllegalArgumentException();
				}
			}
		}
	}

	private static final byte[] MAGIC_STRING = {(byte)'\u0093', 'N', 'U', 'M', 'P', 'Y'};
}