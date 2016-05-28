/*
 * Copyright (c) 2016 Villu Ruusmann
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

import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.dmg.pmml.Row;
import org.jpmml.converter.ValueUtil;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class DOMUtil {

	private DOMUtil(){
	}

	static
	public Row createRow(DocumentBuilder documentBuilder, List<String> keys, List<?> values){
		Row row = new Row();

		Document document = documentBuilder.newDocument();

		if(keys.size() != values.size()){
			throw new IllegalArgumentException();
		}

		for(int i = 0; i < keys.size(); i++){
			Element element = document.createElement(keys.get(i));
			element.setTextContent(ValueUtil.formatValue(values.get(i)));

			row.addContent(element);
		}

		return row;
	}

	static
	public DocumentBuilder createDocumentBuilder(){
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
		documentBuilderFactory.setNamespaceAware(true);

		try {
			return documentBuilderFactory.newDocumentBuilder();
		} catch(Exception e){
			throw new RuntimeException(e);
		}
	}
}