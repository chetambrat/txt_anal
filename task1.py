import gensim
import PyPDF2
import html
import pdfplumber
import xml.etree.cElementTree as ET
from docx import Document
from bs4 import BeautifulSoup as BS
from tika import parser


class Data(object):
    def __init__(self, raw_text):
        self.raw_text = raw_text

    def process(self, type_of_text):
        if type_of_text == "txt":
            with open(self.raw_text, "r", encoding='utf-8') as f:
                data = f.read()
                return data
        elif type_of_text == "html":
            with open(self.raw_text, "r", encoding="utf-8") as f:
                data = f.read()
                raw_html = BS(data, features="lxml").text
                return raw_html
        elif type_of_text == "pdf":
            pdf = pdfplumber.open(self.raw_text)
            page = pdf.pages[0]
            text = page.extract_text()
            pdf.close()
            return text
        elif type_of_text == "docx":
            def getText(filename):
                doc = Document(filename)
                fullText = []
                for para in doc.paragraphs:
                    fullText.append(para.text)
                return '\n'.join(fullText)
            return getText(self.raw_text)


docx = Data("ugolovnie dela sorm rossiushka.docx")
docx_out = docx.process("docx")

pdf = Data("ugolovnie dela sorm rossiushka.pdf")
pdf_out = pdf.process("pdf")

txt = Data("ugolovnie dela sorm rossiushka.txt")
txt_out = txt.process("txt")

html = Data("ugolovnie dela sorm rossiushka.html")
html_out = html.process("html")


def write_xml(input_data, xml_file):
    new_list = []

    article = ET.Element("article")

    new_data = str(input_data)
    splitted = new_data.split("\n")
    for elts in splitted:
        print(elts)
        if elts != "":
            new_list.append(elts)
            paragraph = ET.SubElement(article, "paragraph")
            paragraph.text = str(elts)
            ET.tostring(paragraph, encoding="unicode")

    tree = ET.ElementTree(article)
    tree.write(xml_file)

write_xml(html_out, "result.xml")
write_xml(pdf_out, "result2.xml")
write_xml(txt_out, "result3.xml")
write_xml(docx_out, "resulr4.xml")
