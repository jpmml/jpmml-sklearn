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
package joblib;

import java.io.IOException;
import java.io.InputStream;

import org.jpmml.sklearn.ObjectConstructor;
import org.jpmml.sklearn.Storage;

public class NDArrayWrapperConstructor extends ObjectConstructor {

	private Storage storage = null;


	public NDArrayWrapperConstructor(String module, String name, Storage storage){
		super(module, name, NDArrayWrapper.class);

		setStorage(storage);
	}

	@Override
	public NDArrayWrapper newObject(){
		NDArrayWrapper arrayWrapper = new NDArrayWrapper(getModule(), getName()){

			@Override
			public InputStream getInputStream() throws IOException {
				Storage storage = getStorage();

				return storage.getArray(getFileName());
			}
		};

		return arrayWrapper;
	}

	public Storage getStorage(){
		return this.storage;
	}

	private void setStorage(Storage storage){
		this.storage = storage;
	}
}