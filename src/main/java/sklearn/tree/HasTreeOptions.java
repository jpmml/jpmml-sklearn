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
import java.util.Map;

import org.dmg.pmml.OutputField;
import org.dmg.pmml.tree.Node;
import org.jpmml.converter.HasNativeConfiguration;
import org.jpmml.sklearn.HasSkLearnOptions;
import org.jpmml.sklearn.visitors.TreeModelCompactor;
import org.jpmml.sklearn.visitors.TreeModelFlattener;

public interface HasTreeOptions extends HasSkLearnOptions, HasNativeConfiguration {

	/**
	 * @see TreeModelCompactor
	 */
	String OPTION_COMPACT = "compact";

	/**
	 * @see TreeModelFlattener
	 */
	String OPTION_FLAT = "flat";

	/**
	 * @see Node#hasExtensions()
	 * @see Node#getExtensions()
	 */
	String OPTION_NODE_EXTENSIONS = "node_extensions";

	/**
	 * @see Node#getId()
	 */
	String OPTION_NODE_ID = "node_id";

	/**
	 * @see Node#getScore()
	 */
	String OPTION_NODE_SCORE = "node_score";

	/**
	 * @see OutputField
	 */
	String OPTION_WINNER_ID = "winner_id";

	@Override
	default
	public Map<String, ?> getNativeConfiguration(){
		Map<String, Object> result = new LinkedHashMap<>();
		result.put(HasTreeOptions.OPTION_COMPACT, Boolean.FALSE);
		result.put(HasTreeOptions.OPTION_FLAT, Boolean.FALSE);
		result.put(HasTreeOptions.OPTION_NODE_ID, Boolean.TRUE);
		result.put(HasTreeOptions.OPTION_NODE_SCORE, Boolean.TRUE);
		result.put(HasTreeOptions.OPTION_WINNER_ID, Boolean.FALSE);

		return result;
	}
}