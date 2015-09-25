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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import javax.xml.transform.stream.StreamResult;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import net.razorvine.pickle.objects.ClassDict;
import org.dmg.pmml.PMML;
import org.jpmml.model.JAXBUtil;
import sklearn.Estimator;
import sklearn_pandas.DataFrameMapper;

public class Main {

	@Parameter (
		names = "--pkl-estimator-input",
		description = "Estimator pickle input file",
		required = true
	)
	private File estimatorInput = null;

	@Parameter (
		names = "--pkl-mapper-input",
		description = "DataFrameMapper pickle input file",
		required = false
	)
	private File mapperInput = null;

	@Parameter (
		names = "--pmml-output",
		description = "PMML output file",
		required = true
	)
	private File output = null;


	static
	public void main(String... args) throws Exception {
		Main main = new Main();

		JCommander commander = new JCommander(main);
		commander.setProgramName(Main.class.getName());

		try {
			commander.parse(args);
		} catch(ParameterException pe){
			commander.usage();

			System.exit(-1);
		}

		main.run();
	}

	private void run() throws Exception {
		PMML pmml = convert();

		OutputStream os = new FileOutputStream(this.output);

		try {
			JAXBUtil.marshalPMML(pmml, new StreamResult(os));
		} finally {
			os.close();
		}
	}

	public PMML convert() throws IOException {
		PMML pmml;

		Storage estimatorStorage = PickleUtil.createStorage(this.estimatorInput);

		try {
			Object object = PickleUtil.unpickle(estimatorStorage);

			if(!(object instanceof Estimator)){
				String clazzName = null;

				if(object instanceof ClassDict){
					ClassDict classDict = (ClassDict)object;

					clazzName = (String)classDict.get("__class__");
				}

				throw new IllegalArgumentException("The unpickled estimator object (Python class " + clazzName + ") is not an Estimator or a supported Estimator subclass");
			}

			Estimator estimator = (Estimator)object;

			pmml = estimator.encodePMML();
		} finally {
			estimatorStorage.close();
		}

		if(this.mapperInput != null){
			Storage mapperStorage = PickleUtil.createStorage(this.mapperInput);

			try {
				Object object = PickleUtil.unpickle(mapperStorage);

				if(!(object instanceof DataFrameMapper)){
					throw new IllegalArgumentException("The unpickled mapper object is not a DataFrameMapper");
				}

				DataFrameMapper mapper = (DataFrameMapper)object;

				mapper.updatePMML(pmml);
			} finally {
				mapperStorage.close();
			}
		}

		return pmml;
	}

	public File getEstimatorInput(){
		return this.estimatorInput;
	}

	public void setEstimatorInput(File estimatorInput){
		this.estimatorInput = estimatorInput;
	}

	public File getMapperInput(){
		return this.mapperInput;
	}

	public void setMapperInput(File mapperInput){
		this.mapperInput = mapperInput;
	}

	public File getOutput(){
		return this.output;
	}

	public void setOutput(File output){
		this.output = output;
	}
}