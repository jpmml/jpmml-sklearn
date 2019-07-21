package numpy.core;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;

import com.google.common.io.ByteStreams;
import com.google.common.primitives.UnsignedInts;
import org.dmg.pmml.DataType;

/**
 * http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
 * http://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.byteorder.html
 */
public class TypeDescriptor {

	private String descr = null;

	private ByteOrder byteOrder = null;

	private TypeDescriptor.Kind kind = null;

	private int size = 0;


	public TypeDescriptor(String descr){
		setDescr(descr);

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

		TypeDescriptor.Kind kind = Kind.forChar(descr.charAt(i));

		i++;

		setKind(kind);

		if(i < descr.length()){
			int size = Integer.parseInt(descr.substring(i));

			setSize(size);
		}
	}

	public DataType getDataType(){
		String descr = getDescr();
		TypeDescriptor.Kind kind = getKind();
		int size = getSize();

		switch(kind){
			case BOOLEAN:
				return DataType.BOOLEAN;
			case INTEGER:
			case UNSIGNED_INTEGER:
				return DataType.INTEGER;
			case FLOAT:
				switch(size){
					case 4:
						return DataType.FLOAT;
					case 8:
						return DataType.DOUBLE;
					default:
						throw new IllegalArgumentException(descr);
				}
			case STRING:
			case UNICODE:
				return DataType.STRING;
			default:
				throw new IllegalArgumentException(descr);
		}
	}

	public Object read(InputStream is) throws IOException {
		String descr = getDescr();
		TypeDescriptor.Kind kind = getKind();
		ByteOrder byteOrder = getByteOrder();
		int size = getSize();

		switch(kind){
			case BOOLEAN:
				{
					switch(size){
						case 1:
							return (NDArrayUtil.readByte(is) == 1);
						default:
							break;
					}
				}
				break;
			case INTEGER:
				{
					switch(size){
						case 1:
							return NDArrayUtil.readByte(is);
						case 2:
							return NDArrayUtil.readShort(is, byteOrder);
						case 4:
							return NDArrayUtil.readInt(is, byteOrder);
						case 8:
							return NDArrayUtil.readLong(is, byteOrder);
						default:
							break;
					}
				}
				break;
			case UNSIGNED_INTEGER:
				{
					switch(size){
						case 1:
							return NDArrayUtil.readUnsignedByte(is);
						case 2:
							return NDArrayUtil.readUnsignedShort(is, byteOrder);
						case 4:
							return UnsignedInts.toLong(NDArrayUtil.readInt(is, byteOrder));
						case 8:
							String string = Long.toUnsignedString(NDArrayUtil.readLong(is, byteOrder));

							return Long.parseUnsignedLong(string);
						default:
							break;
					}
				}
				break;
			case FLOAT:
				{
					switch(size){
						case 4:
							return NDArrayUtil.readFloat(is, byteOrder);
						case 8:
							return NDArrayUtil.readDouble(is, byteOrder);
						default:
							break;
					}
				}
				break;
			case OBJECT:
				{
					return NDArrayUtil.readObject(is);
				}
			case STRING:
				{
					return NDArrayUtil.readString(is, size);
				}
			case UNICODE:
				{
					return NDArrayUtil.readUnicode(is, byteOrder, size);
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

		throw new IllegalArgumentException(descr);
	}

	public boolean isObject(){
		TypeDescriptor.Kind kind = getKind();

		switch(kind){
			case OBJECT:
				return true;
			default:
				return false;
		}
	}

	public String getDescr(){
		return this.descr;
	}

	private void setDescr(String descr){
		this.descr = descr;
	}

	public ByteOrder getByteOrder(){
		return this.byteOrder;
	}

	private void setByteOrder(ByteOrder byteOrder){
		this.byteOrder = byteOrder;
	}

	public TypeDescriptor.Kind getKind(){
		return this.kind;
	}

	private void setKind(TypeDescriptor.Kind kind){
		this.kind = kind;
	}

	public int getSize(){
		return this.size;
	}

	private void setSize(int size){
		this.size = size;
	}

	static
	public enum Kind {
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
		public TypeDescriptor.Kind forChar(char c){

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