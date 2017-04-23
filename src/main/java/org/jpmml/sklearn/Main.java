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
import java.util.Collections;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.Estimator;
import sklearn.pipeline.Pipeline;
import sklearn2pmml.PMMLPipeline;

public class Main {

	@Parameter (
		names = "--help",
		description = "Show the list of configuration options and exit",
		help = true
	)
	private boolean help = false;

	@Parameter (
		names = {"--pkl-pipeline-input", "--pkl-input"},
		description = "Pickle input file",
		required = true
	)
	private File input = null;

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

	public void run() throws Exception {
		Object object;

		try(Storage storage = PickleUtil.createStorage(this.input)){
			logger.info("Parsing PKL..");

			long start = System.currentTimeMillis();
			object = PickleUtil.unpickle(storage);
			long end = System.currentTimeMillis();

			logger.info("Parsed PKL in {} ms.", (end - start));
		} catch(Exception e){
			logger.error("Failed to parse PKL", e);

			throw e;
		}

		if(!(object instanceof PMMLPipeline)){

			// Create a single- or multi-step PMMLPipeline from a Pipeline
			if(object instanceof Pipeline){
				Pipeline pipeline = (Pipeline)object;

				object = new PMMLPipeline()
					.setSteps(pipeline.getSteps());
			} else

			// Create a single-step PMMLPipeline from an Estimator
			if(object instanceof Estimator){
				Estimator estimator = (Estimator)object;

				object = new PMMLPipeline()
					.setSteps(Collections.<Object[]>singletonList(new Object[]{"estimator", estimator}));
			} else

			{
				throw new IllegalArgumentException("The object (" + ClassDictUtil.formatClass(object) + ") is not a PMMLPipeline");
			}
		}

		PMMLPipeline pipeline = (PMMLPipeline)object;

		PMML pmml;

		try {
			logger.info("Converting..");

			long begin = System.currentTimeMillis();
			pmml = pipeline.encodePMML();
			long end = System.currentTimeMillis();

			logger.info("Converted in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to convert", e);

			throw e;
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

	public File getInput(){
		return this.input;
	}

	public void setInput(File input){
		this.input = input;
	}

	public File getOutput(){
		return this.output;
	}

	public void setOutput(File output){
		this.output = output;
	}

	private static final Logger logger = LoggerFactory.getLogger(Main.class);
}