/*
 * Copyright (c) 2024 Villu Ruusmann
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
package sklearn2pmml;

import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.ParameterizedType;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import javax.xml.XMLConstants;
import javax.xml.namespace.NamespaceContext;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathFactory;

import com.google.common.collect.Iterables;
import jakarta.xml.bind.Binder;
import jakarta.xml.bind.JAXBContext;
import jakarta.xml.bind.Marshaller;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.JAXBSerializer;
import org.jpmml.model.JAXBUtil;
import org.jpmml.model.ReflectionUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sklearn.SkLearnException;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.xml.sax.Locator;

public class CustomizationUtil {

	private CustomizationUtil(){
	}

	static
	public void customize(Model model, List<? extends Customization> customizations) throws Exception {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
		documentBuilderFactory.setNamespaceAware(true);

		XPathFactory xPathFactory = XPathFactory.newInstance();

		DocumentBuilder documentBuilder = documentBuilderFactory.newDocumentBuilder();

		Document document = documentBuilder.newDocument();

		NamespaceContext namespaceContext = new DocumentNamespaceContext(document);

		JAXBContext context = JAXBUtil.getContext();

		Binder<Node> binder = context.createBinder(Node.class);
		binder.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, Boolean.FALSE);
		binder.setProperty(Marshaller.JAXB_FRAGMENT, Boolean.TRUE);

		binder.marshal(model, document);

		for(Customization customization : customizations){
			String command = customization.getCommand();

			PMMLObject xPathExprObject;

			switch(command){
				case Customization.COMMAND_INSERT:
				case Customization.COMMAND_UPDATE:
					{
						String xPathExpr = customization.getOptionalXPathExpr();

						if(xPathExpr == null){
							xPathExprObject = model;

							break;
						}
					}
					// Falls through
				case Customization.COMMAND_DELETE:
					{
						String xPathExpr = customization.getXPathExpr();

						XPath xPath = xPathFactory.newXPath();
						xPath.setNamespaceContext(namespaceContext);

						XPathExpression xPathExpression = xPath.compile(xPathExpr);

						Node node = (Node)xPathExpression.evaluate(document.getDocumentElement(), XPathConstants.NODE);

						xPathExprObject = (PMMLObject)binder.getJAXBNode(node);
						if(xPathExprObject == null){
							throw new SkLearnException("XPath expression \'" + xPathExpr + "\' is not associated with a PMML object");
						}
					}
					break;
				default:
					throw new IllegalArgumentException(command);
			}

			PMMLObject pmmlElementObject;

			switch(command){
				case Customization.COMMAND_INSERT:
				case Customization.COMMAND_UPDATE:
					{
						String pmmlElement = customization.getPMMLElement();

						pmmlElementObject = (PMMLObject)parsePMML(pmmlElement);
					}
					break;
				case Customization.COMMAND_DELETE:
					{
						pmmlElementObject = null;
					}
					break;
				default:
					throw new IllegalArgumentException(command);
			}

			PMMLObject object;

			switch(command){
				case Customization.COMMAND_INSERT:
					{
						object = insert(xPathExprObject, pmmlElementObject);
					}
					break;
				case Customization.COMMAND_UPDATE:
					{
						object = update(xPathExprObject, pmmlElementObject);
					}
					break;
				case Customization.COMMAND_DELETE:
					{
						Set<PMMLObject> parents = new HashSet<>();

						Visitor parentFinder = new AbstractVisitor(){

							@Override
							public VisitorAction visit(PMMLObject object){

								if(Objects.equals(xPathExprObject, object)){
									PMMLObject parent = getParent();

									parents.add(parent);

									return VisitorAction.TERMINATE;
								}

								return super.visit(object);
							}
						};
						parentFinder.applyTo(model);

						if(parents.size() != 1){
							throw new IllegalArgumentException();
						}

						object = delete(Iterables.getOnlyElement(parents), xPathExprObject);
					}
					break;
				default:
					throw new IllegalArgumentException(command);
			}

			if(object != null){
				Node node = binder.getXMLNode(object);

				// XXX: The node is always null!?
				if(node != null){
					binder.updateXML(object, node);
				}
			}
		}
	}

	static
	private PMMLObject insert(PMMLObject parent, PMMLObject child) throws ReflectiveOperationException {
		Field field = findField(parent, child);

		Class<?> fieldType = field.getType();

		if(Objects.equals(List.class, fieldType)){
			addListElement(field, parent, child);
		} else

		{
			PMMLObject fieldValue = ReflectionUtil.getFieldValue(field, parent);
			if(fieldValue != null){
				throw new IllegalArgumentException();
			}

			ReflectionUtil.setFieldValue(field, parent, child);
		}

		return parent;
	}

	static
	private PMMLObject update(PMMLObject target, PMMLObject source){
		Class<? extends PMMLObject> targetClazz = target.getClass();
		Class<? extends PMMLObject> sourceClazz = source.getClass();

		if(!Objects.equals(targetClazz, sourceClazz)){
			throw new IllegalArgumentException();
		}

		List<Field> fields = ReflectionUtil.getFields(targetClazz);
		for(Field field : fields){
			Class<?> fieldType = field.getType();

			if(Objects.equals(Locator.class, fieldType)){
				continue;
			}

			Object sourceValue = ReflectionUtil.getFieldValue(field, source);
			if(sourceValue == null){
				continue;
			}

			ReflectionUtil.setFieldValue(field, target, sourceValue);
		}

		return target;
	}

	static
	private PMMLObject delete(PMMLObject parent, PMMLObject child){
		Field field = findField(parent, child);

		Class<?> fieldType = field.getType();

		if(Objects.equals(List.class, fieldType)){
			removeListElement(field, parent, child);
		} else

		{
			ReflectionUtil.setFieldValue(field, parent, null);
		}

		return parent;
	}

	static
	private Field findField(PMMLObject parent, PMMLObject child){
		Class<? extends PMMLObject> parentClazz = parent.getClass();
		Class<? extends PMMLObject> childClazz = child.getClass();

		List<Field> fields = ReflectionUtil.getFields(parentClazz);

		for(Field field : fields){
			Class<?> fieldType = field.getType();

			if(Objects.equals(List.class, fieldType)){
				ParameterizedType listType = (ParameterizedType)field.getGenericType();

				Class<?> listElementType = (Class<?>)listType.getActualTypeArguments()[0];

				if((PMMLObject.class).isAssignableFrom(listElementType) && listElementType.isAssignableFrom(childClazz)){
					return field;
				}
			} else

			{
				if((PMMLObject.class).isAssignableFrom(fieldType) && fieldType.isAssignableFrom(childClazz)){
					return field;
				}
			}
		}

		throw new IllegalArgumentException();
	}

	static
	private void addListElement(Field field, PMMLObject parent, PMMLObject child) throws ReflectiveOperationException {
		@SuppressWarnings("unused")
		List<?> fieldValue = (List<?>)ReflectionUtil.getFieldValue(field, parent);

		ParameterizedType listType = (ParameterizedType)field.getGenericType();

		Class<?> listElementType = (Class<?>)listType.getActualTypeArguments()[0];

		Method appenderMethod = ReflectionUtil.getAppenderMethod(field);

		// See https://stackoverflow.com/a/36125994
		Object[] valueArray = (Object[])Array.newInstance(listElementType, 1);
		valueArray[0] = child;

		appenderMethod.invoke(parent, (Object)valueArray);
	}

	static
	private void removeListElement(Field field, PMMLObject parent, PMMLObject child){
		List<?> fieldValue = (List<?>)ReflectionUtil.getFieldValue(field, parent);

		boolean success = fieldValue.remove(child);
		if(!success){
			throw new IllegalArgumentException();
		}
	}

	static
	public Object parsePMML(String string){

		try(Reader reader = new StringReader(string)){
			JAXBSerializer jaxbSerializer = new JAXBSerializer();

			return jaxbSerializer.unmarshal(new StreamSource(reader));
		} catch(Exception e){
			throw new SkLearnException("Failed to parse PMML string", e);
		}
	}

	static
	public String formatPMML(PMMLObject object){

		try(StringWriter writer = new StringWriter()){
			JAXBSerializer jaxbSerializer = new JAXBSerializer();

			jaxbSerializer.marshal(object, new StreamResult(writer));

			return writer.toString();
		} catch(Exception e){
			throw new SkLearnException("Failed to format PMML object", e);
		}
	}

	static
	private class DocumentNamespaceContext implements NamespaceContext {

		private Document document = null;


		private DocumentNamespaceContext(Document document){
			setDocument(document);
		}

		@Override
		public String getNamespaceURI(String prefix){
			Document document = getDocument();

			if(Objects.equals(XMLConstants.DEFAULT_NS_PREFIX, prefix)){
				return document.lookupNamespaceURI(null);
			}

			return document.lookupNamespaceURI(prefix);
		}

		@Override
		public String getPrefix(String namespaceURI){
			Document document = getDocument();

			return document.lookupPrefix(namespaceURI);
		}

		@Override
		public Iterator<String> getPrefixes(String namespaceURI){
			throw new UnsupportedOperationException();
		}

		public Document getDocument(){
			return this.document;
		}

		private void setDocument(Document document){
			this.document = document;
		}
	}
}