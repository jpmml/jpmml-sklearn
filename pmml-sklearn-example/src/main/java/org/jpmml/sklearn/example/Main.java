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
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.dmg.pmml.PMML;
import org.jpmml.model.metro.MetroJAXBUtil;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.Storage;
import org.jpmml.python.StorageUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sklearn.tree.HasTreeOptions;

public class Main {

	@Parameter (
		names = {"--help"},
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
		names = {"--pmml-output"},
		description = "PMML output file",
		required = true
	)
	private File output = null;

	@Parameter (
		names = {"--X-" + HasTreeOptions.OPTION_COMPACT},
		description = "Transform SkLearn-style trees to PMML-style trees",
		arity = 1
	)
	private Boolean compact = null;

	@Parameter (
		names = {"--X-" + HasTreeOptions.OPTION_FLAT},
		description = "Flatten trees",
		arity = 1
	)
	private Boolean flat = null;

	@Parameter (
		names = {"--X-" + HasTreeOptions.OPTION_NODE_ID},
		description = "Keep SkLearn node identifiers",
		arity = 1
	)
	private Boolean nodeId = null;

	@Parameter (
		names = {"--X-" + HasTreeOptions.OPTION_NODE_SCORE},
		description = "Keep SkLearn node scores for branch (non-leaf) nodes",
		arity = 1
	)
	private Boolean nodeScore = null;

	@Parameter (
		names = {"--X-" + HasTreeOptions.OPTION_NUMERIC},
		description = "Transform non-numeric node split conditions to numeric",
		arity = 1
	)
	private Boolean numeric = null;

	@Parameter (
		names = {"--X-" + HasTreeOptions.OPTION_PRUNE},
		description = "Truncate invariant leaf nodes",
		arity = 1
	)
	private Boolean prune = null;

	@Parameter (
		names = {"--X-" + HasTreeOptions.OPTION_WINNER_ID},
		description = "Output node identifiers",
		arity = 1
	)
	private Boolean winnerId = null;


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
		SkLearnEncoder encoder = new SkLearnEncoder();

		Object object;

		try(Storage storage = StorageUtil.createStorage(this.input)){
			logger.info("Parsing PKL..");

			long begin = System.currentTimeMillis();
			object = PickleUtil.unpickle(storage);
			long end = System.currentTimeMillis();

			logger.info("Parsed PKL in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to parse PKL", e);

			throw e;
		}

		Encodable encodable = EncodableUtil.toEncodable(object);

		Map<String, ?> options = getOptions();
		if(!options.isEmpty()){
			encodable.setPMMLOptions(options);
		}

		PMML pmml;

		try {
			logger.info("Converting PKL to PMML..");

			long begin = System.currentTimeMillis();
			pmml = encodable.encodePMML(encoder);
			long end = System.currentTimeMillis();

			logger.info("Converted PKL to PMML in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to convert PKL to PMML", e);

			throw e;
		}

		try(OutputStream os = new FileOutputStream(this.output)){
			logger.info("Marshalling PMML..");

			long begin = System.currentTimeMillis();
			MetroJAXBUtil.marshalPMML(pmml, os);
			long end = System.currentTimeMillis();

			logger.info("Marshalled PMML in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to marshal PMML", e);

			throw e;
		}
	}

	private Map<String, ?> getOptions(){
		Map<String, Object> options = new LinkedHashMap<>();

		options.put(HasTreeOptions.OPTION_COMPACT, this.compact);
		options.put(HasTreeOptions.OPTION_FLAT, this.flat);
		options.put(HasTreeOptions.OPTION_NODE_ID, this.nodeId);
		options.put(HasTreeOptions.OPTION_NODE_SCORE, this.nodeScore);
		options.put(HasTreeOptions.OPTION_NUMERIC, this.numeric);
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

	private static final Logger logger = LoggerFactory.getLogger(Main.class);
}