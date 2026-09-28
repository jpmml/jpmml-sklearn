/*
 * Copyright (c) 2017 Villu Ruusmann
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
package sklearn.tree;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.OutputField;
import org.dmg.pmml.tree.Node;
import org.jpmml.converter.HasSchemaOptions;
import org.jpmml.converter.tree.HasTreeModelOptions;
import org.jpmml.sklearn.HasSkLearnOptions;

public interface HasTreeOptions extends HasSkLearnOptions, HasSchemaOptions, HasTreeModelOptions {

	/**
	 * @see Node#getDefaultChild()
	 */
	String OPTION_ALLOW_MISSING = "allow_missing";

	/**
	 * @see Node#hasExtensions()
	 * @see Node#getExtensions()
	 */
	String OPTION_NODE_EXTENSIONS = "node_extensions";

	/**
	 * @see Node#getScore()
	 */
	String OPTION_NODE_SCORE = "node_score";

	/**
	 * @see OutputField
	 */
	String OPTION_WINNER_ID = "winner_id";

	List<String> OPTIONS_NODE = List.of(HasTreeOptions.OPTION_NODE_EXTENSIONS, HasTreeOptions.OPTION_NODE_ID, HasTreeOptions.OPTION_NODE_RECORDCOUNT, HasTreeOptions.OPTION_NODE_SCORE, HasTreeOptions.OPTION_WINNER_ID);

	@Override
	default
	public Map<String, ?> getNativeConfiguration(){
		Map<String, Object> result = new LinkedHashMap<>();
		result.putAll(HasTreeModelOptions.super.getNativeConfiguration());
		result.put(HasTreeOptions.OPTION_INPUT_FLOAT, Boolean.TRUE);
		result.put(HasTreeOptions.OPTION_NODE_ID, Boolean.TRUE);
		result.put(HasTreeOptions.OPTION_NODE_SCORE, Boolean.TRUE);
		result.put(HasTreeOptions.OPTION_NUMERIC, Boolean.TRUE);
		result.put(HasTreeOptions.OPTION_WINNER_ID, Boolean.FALSE);

		return result;
	}
}