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
import java.io.OutputStream;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.dmg.pmml.Extension;
import org.dmg.pmml.MiningBuildTask;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Estimator;
import sklearn.EstimatorUtil;
import sklearn_pandas.DataFrameMapper;

public class Main {

	@Parameter (
		names = {"--pkl-input", "--pkl-estimator-input"},
		description = "Estimator pickle input file",
		required = true
	)
	private File estimatorInput = null;

	@Parameter (
		names = "--repr-estimator",
		description = "Estimator string representation",
		hidden = true
	)
	private String estimatorRepr = null;

	@Parameter (
		names = "--help",
		description = "Show the list of configuration options and exit",
		help = true
	)
	private boolean help = false;

	@Parameter (
		names = "--pkl-mapper-input",
		description = "DataFrameMapper pickle input file",
		required = false
	)
	private File mapperInput = null;

	@Parameter (
		names = "--repr-mapper",
		description = "DataFrameMapper string representation",
		hidden = true
	)
	private String mapperRepr = null;

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
			StringBuilder sb = new StringBuilder();

			sb.append(pe.toString());
			sb.append("\n");

			commander.usage(sb);

			System.err.println(sb.toString());

			System.exit(-1);
		}

		if(main.help){
			StringBuilder sb = new StringBuilder();

			commander.usage(sb);

			System.out.println(sb.toString());

			System.exit(0);
		}

		main.run();
	}

	private void run() throws Exception {
		PMML pmml;

		FeatureMapper featureMapper = new FeatureMapper();

		Map<String, String> reprs = new LinkedHashMap<>();

		if(this.mapperInput != null){

			try(Storage storage = PickleUtil.createStorage(this.mapperInput)){
				Object object;

				try {
					logger.info("Parsing DataFrameMapper PKL..");

					long start = System.currentTimeMillis();
					object = PickleUtil.unpickle(storage);
					long end = System.currentTimeMillis();

					logger.info("Parsed DataFrameMapper PKL in {} ms.", (end - start));
				} catch(Exception e){
					logger.error("Failed to parse DataFrameMapper PKL", e);

					throw e;
				}

				if(!(object instanceof DataFrameMapper)){
					throw new IllegalArgumentException("The mapper object (" + ClassDictUtil.formatClass(object) + ") is not a DataFrameMapper");
				}

				DataFrameMapper mapper = (DataFrameMapper)object;

				try {
					logger.info("Converting DataFrameMapper..");

					long start = System.currentTimeMillis();
					mapper.encodeFeatures(featureMapper);
					long end = System.currentTimeMillis();

					logger.info("Converted DataFrameMapper in {} ms.", (end - start));
				} catch(Exception e){
					logger.error("Failed to convert DataFrameMapper", e);

					throw e;
				}
			}

			if(this.mapperRepr != null){
				reprs.put("mapper", this.mapperRepr);
			}
		}

		try(Storage storage = PickleUtil.createStorage(this.estimatorInput)){
			Object object;

			try {
				logger.info("Parsing Estimator PKL..");

				long start = System.currentTimeMillis();
				object = PickleUtil.unpickle(storage);
				long end = System.currentTimeMillis();

				logger.info("Parsed Estimator PKL in {} ms.", (end - start));
			} catch(Exception e){
				logger.error("Failed to parse Estimator PKL", e);

				throw e;
			}

			if(!(object instanceof Estimator)){
				throw new IllegalArgumentException("The estimator object (" + ClassDictUtil.formatClass(object) + ") is not an Estimator or is not a supported Estimator subclass");
			}

			Estimator estimator = (Estimator)object;

			try {
				logger.info("Converting Estimator..");

				long start = System.currentTimeMillis();
				pmml = EstimatorUtil.encodePMML(estimator, featureMapper);
				long end = System.currentTimeMillis();

				logger.info("Converted Estimator in {} ms.", (end - start));
			} catch(Exception e){
				logger.error("Failed to convert Estimator", e);

				throw e;
			}
		}

		if(this.estimatorRepr != null){
			reprs.put("estimator", this.estimatorRepr);
		}

		Collection<Map.Entry<String, String>> entries = reprs.entrySet();
		for(Map.Entry<String, String> entry : entries){
			addObjectRepr(pmml, entry.getKey(), entry.getValue());
		}

		try(OutputStream os = new FileOutputStream(this.output)){
			logger.info("Marshalling PMML..");

			long start = System.currentTimeMillis();
			MetroJAXBUtil.marshalPMML(pmml, os);
			long end = System.currentTimeMillis();

			logger.info("Marshalled PMML in {} ms.", (end - start));
		} catch(Exception e){
			logger.error("Failed to marshal PMML", e);

			throw e;
		}
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

	static
	private void addObjectRepr(PMML pmml, String name, String content){
		MiningBuildTask miningBuildTask = pmml.getMiningBuildTask();

		if(miningBuildTask == null){
			miningBuildTask = new MiningBuildTask();

			pmml.setMiningBuildTask(miningBuildTask);
		}

		Extension extension = new Extension()
			.setName(name)
			.setValue("repr(" + name + ")")
			.addContent(content);

		miningBuildTask.addExtensions(extension);
	}

	private static final Logger logger = LoggerFactory.getLogger(Main.class);
}