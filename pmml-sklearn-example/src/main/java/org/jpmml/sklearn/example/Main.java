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
package org.jpmml.sklearn.example;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.logging.LogManager;

import com.beust.jcommander.DefaultUsageFormatter;
import com.beust.jcommander.IUsageFormatter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.dmg.pmml.PMML;
import org.jpmml.converter.LoggingUtil;
import org.jpmml.converter.PMMLEncodable;
import org.jpmml.converter.Parameters;
import org.jpmml.converter.SchemaParameters;
import org.jpmml.converter.tree.TreeModelParameters;
import org.jpmml.model.JAXBSerializer;
import org.jpmml.model.metro.MetroJAXBSerializer;
import org.jpmml.python.PythonUnpickler;
import org.jpmml.python.Storage;
import org.jpmml.python.StorageUtil;
import org.jpmml.sklearn.SkLearnUnpickler;
import org.jpmml.sklearn.SkLearnUtil;
import org.jpmml.telemetry.Incident;
import org.jpmml.telemetry.TelemetryClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.HasRegressorOptions;
import sklearn.tree.HasTreeOptions;
import sklearn2pmml.statsmodels.StatsModelsUtil;

public class Main {

	@Parameter (
		names = {"--pkl-pipeline-input", "--pkl-input"},
		description = "Pickle input file",
		required = true,
		order = Parameters.BLOCK_IO + 1
	)
	private File input = null;

	@Parameter (
		names = {Parameters.NAME_PMML_OUTPUT},
		description = Parameters.DESCRIPTION_PMML_OUTPUT,
		required = true,
		order = Parameters.BLOCK_IO + 2
	)
	private File output = null;

	@Parameter (
		names = {SchemaParameters.NAME_INPUT_FLOAT},
		description = SchemaParameters.DESCRIPTION_INPUT_FLOAT,
		arity = 1,
		order = Parameters.BLOCK_SCHEMA_OPTIONS + 1
	)
	private Boolean inputFloat = null;

	@Parameter (
		names = {SchemaParameters.NAME_NUMERIC},
		description = SchemaParameters.DESCRIPTION_NUMERIC,
		arity = 1,
		order = Parameters.BLOCK_SCHEMA_OPTIONS + 2
	)
	private Boolean numeric = null;

	@Parameter (
		names = {Parameters.PREFIX_OPTION + HasTreeOptions.OPTION_ALLOW_MISSING},
		description = "Allow missing-value-aware splits",
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 1
	)
	private Boolean allowMissing = null;

	@Parameter (
		names = {TreeModelParameters.NAME_COMPACT},
		description = TreeModelParameters.DESCRIPTION_COMPACT,
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 2
	)
	private Boolean compact = null;

	@Parameter (
		names = {Parameters.PREFIX_OPTION + HasRegressorOptions.OPTION_CONFIDENCE_LEVEL},
		description = "Output prediction interval bounds at the given confidence level",
		converter = ConfidenceLevelConverter.class,
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 3
	)
	private ConfidenceLevel confidenceLevel = null;

	@Parameter (
		names = {TreeModelParameters.NAME_FLAT},
		description = TreeModelParameters.DESCRIPTION_FLAT,
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 4
	)
	private Boolean flat = null;

	@Parameter (
		names = {TreeModelParameters.NAME_NODE_ID},
		description = TreeModelParameters.DESCRIPTION_NODE_ID,
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 5
	)
	private Boolean nodeId = null;

	@Parameter (
		names = {TreeModelParameters.NAME_NODE_RECORDCOUNT},
		description = TreeModelParameters.DESCRIPTION_NODE_RECORDCOUNT,
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 6
	)
	private Boolean nodeRecordCount = null;

	@Parameter (
		names = {Parameters.PREFIX_OPTION + HasTreeOptions.OPTION_NODE_SCORE},
		description = "Keep branch node scores",
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 7
	)
	private Boolean nodeScore = null;

	@Parameter (
		names = {TreeModelParameters.NAME_PRUNE},
		description = TreeModelParameters.DESCRIPTION_PRUNE,
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 8
	)
	private Boolean prune = null;

	@Parameter (
		names = {Parameters.PREFIX_OPTION + HasTreeOptions.OPTION_WINNER_ID},
		description = "Output winning node identifiers",
		arity = 1,
		order = Parameters.BLOCK_MODEL_OPTIONS + 9
	)
	private Boolean winnerId = null;

	@Parameter (
		names = {Parameters.NAME_HELP},
		description = Parameters.DESCRIPTION_HELP,
		help = true,
		order = Integer.MAX_VALUE
	)
	private boolean help = false;


	static
	public void main(String... args){
		Main main = new Main();

		JCommander commander = new JCommander(main);
		commander.setProgramName(Main.class.getName());

		IUsageFormatter usageFormatter = new DefaultUsageFormatter(commander);

		try {
			commander.parse(args);
		} catch(ParameterException pe){
			StringBuilder sb = new StringBuilder();

			sb.append(pe.toString());
			sb.append("\n");

			usageFormatter.usage(sb);

			System.err.println(sb.toString());

			System.exit(-1);
		}

		if(main.help){
			StringBuilder sb = new StringBuilder();

			usageFormatter.usage(sb);

			System.out.println(sb.toString());

			System.exit(0);
		}

		try {
			main.run();
		} catch(FileNotFoundException fnfe){
			System.exit(-1);
		} catch(Exception e){
			Package _package = Main.class.getPackage();

			Map<String, String> environment = new LinkedHashMap<>();
			environment.put("jpmml-sklearn", _package.getImplementationVersion());

			Incident incident = new Incident()
				.setProject("jpmml-sklearn")
				.setEnvironment(environment)
				.setException(e);

			try {
				TelemetryClient.report("https://telemetry.jpmml.org/v1/incidents", incident);
			} catch(InterruptedException ie){
				// Ignored
			} catch(IOException ioe){
				// Ignored
			}

			System.exit(-1);
		}
	}

	public void run() throws Exception {
		Object object;

		try(Storage storage = StorageUtil.createStorage(this.input)){
			logger.info("Parsing PKL..");

			PythonUnpickler pythonUnpickler = new SkLearnUnpickler();

			long begin = System.currentTimeMillis();
			object = pythonUnpickler.load(storage);
			long end = System.currentTimeMillis();

			logger.info("Parsed PKL in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to parse PKL", e);

			throw e;
		}

		PMMLEncodable encodable;

		try {
			encodable = SkLearnUtil.toEncodable(object);
		} catch(Exception e){
			logger.error("Failed to recognize PKL", e);

			throw e;
		}

		Map<String, ?> options = getOptions();

		PMML pmml;

		try {
			logger.info("Converting PKL to PMML..");

			long begin = System.currentTimeMillis();
			pmml = SkLearnUtil.encodePMML(encodable, options);
			long end = System.currentTimeMillis();

			logger.info("Converted PKL to PMML in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to convert PKL to PMML", e);

			throw e;
		}

		try(OutputStream os = new FileOutputStream(this.output)){
			logger.info("Marshalling PMML..");

			JAXBSerializer jaxbSerializer = new MetroJAXBSerializer();

			long begin = System.currentTimeMillis();
			jaxbSerializer.serializePretty(pmml, os);
			long end = System.currentTimeMillis();

			logger.info("Marshalled PMML in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to marshal PMML", e);

			throw e;
		}
	}

	private Map<String, ?> getOptions(){
		Map<String, Object> options = new LinkedHashMap<>();

		Object confidenceLevel = (this.confidenceLevel != null ? this.confidenceLevel.getValue() : null);

		options.put(HasTreeOptions.OPTION_INPUT_FLOAT, this.inputFloat);
		options.put(HasTreeOptions.OPTION_NUMERIC, this.numeric);
		options.put(HasTreeOptions.OPTION_ALLOW_MISSING, this.allowMissing);
		options.put(HasTreeOptions.OPTION_COMPACT, this.compact);
		options.put(HasRegressorOptions.OPTION_CONFIDENCE_LEVEL, confidenceLevel);
		options.put(HasTreeOptions.OPTION_FLAT, this.flat);
		options.put(HasTreeOptions.OPTION_NODE_ID, this.nodeId);
		options.put(HasTreeOptions.OPTION_NODE_RECORDCOUNT, this.nodeRecordCount);
		options.put(HasTreeOptions.OPTION_NODE_SCORE, this.nodeScore);
		options.put(HasTreeOptions.OPTION_PRUNE, this.prune);
		options.put(HasTreeOptions.OPTION_WINNER_ID, this.winnerId);

		// Ignore defaults
		options.values().removeIf(Objects::isNull);

		return options;
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

	static {
		LogManager logManager = LogManager.getLogManager();
		logManager.reset();

		LoggingUtil.configureConsole();
	}

	private static final Logger logger = LoggerFactory.getLogger(Main.class);

	static {
		SkLearnUtil.initOnce();
		StatsModelsUtil.initOnce();
	}
}