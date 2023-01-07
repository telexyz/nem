import mugrade

with open("pds.html", encoding="utf-8") as f:
    course_page_xml = f.read()
    
with open("cmu.html", encoding="utf-8") as f:
    cmu_page_xml = f.read()
    
    
test_xml = """<?xml version="1.0" encoding="UTF-8"?>
<!-- This is a comment -->
<note date="8/31/12">
    <to>Tove</to>
    <from>Jani</from>
    <heading type="Reminder"/>
    <body>Don't forget me this weekend!</body>
    <!-- This is a multiline comment,
         which take a bit of care to parse -->
</note>"""


def to_string(xml_node):
    """ Simple function to convert XML tree to a string (without nice formatting)"""
    attr_string = ' '.join([k + '="' + v +'"' for k,v in xml_node.attributes.items()])
    string = f"<{xml_node.tag}{(' ' + attr_string) if len(attr_string) > 0 else ''}>"
    for child in xml_node.children:
        string += to_string(child)
    if len(xml_node.children) == 0:
        string += xml_node.content
    string += f"</{xml_node.tag}>"
    return string



def test_tag_regex():
    with mugrade.test: assert tag_regex(test_xml) == {'tag_open': [('note', ' date="8/31/12"'), ('to', ''), ('from', ''), ('heading', ' type="Reminder"/'),  ('body', '')],
 'tag_close': ['to', 'from', 'body', 'note'],
 'tag_open_close': [('heading', ' type="Reminder"')],
 'comment': [' This is a comment ', ' This is a multiline comment,\n         which take a bit of care to parse '],
 'xml_prolog': ['xml version="1.0" encoding="UTF-8"'],
 'html_declaration': []}
    with mugrade.test: assert {k:len(v) for k,v in tag_regex(course_page_xml).items()} == {'tag_open': 74, 'tag_close': 60, 'tag_open_close': 14, 'comment': 2, 'xml_prolog': 0, 'html_declaration': 1}

        
def submit_tag_regex():
    mugrade.submit({k:len(v) for k,v in tag_regex(cmu_page_xml).items()})
    mugrade.submit(tag_regex(cmu_page_xml)["tag_open_close"])
    
    
def test_create_xml_tree():
    with mugrade.test: assert create_xml_tree('<?xml version="1.0" encoding="UTF-8"?>').children == []
    with mugrade.test: assert create_xml_tree('<body>Lorem Ipsum Dolor Sit Amet</body>').children[0].tag == "body"
    with mugrade.test: assert create_xml_tree("<body class='A' attr = ' b' foo='bar'   />").children[0].tag == "body"
    with mugrade.test: assert create_xml_tree("<body class='A' attr = ' b' foo='bar'   />").children[0].attributes == {"class":"A", "attr":" b", "foo":"bar"}
    with mugrade.test: assert to_string(create_xml_tree(test_xml)) == '<><note date="8/31/12"><to>Tove</to><from>Jani</from><heading type="Reminder"></heading><body>Don\'t forget me this weekend!</body></note></>'
        

def submit_create_xml_tree():
    mugrade.submit(to_string(create_xml_tree(course_page_xml).children[0].children[0]))
    mugrade.submit(to_string(create_xml_tree(cmu_page_xml).children[0].children[0]))
    
    
def test_create_searchable_xml_tree():
    test_xml_tree = create_searchable_xml_tree(test_xml)
    with mugrade.test: assert len(test_xml_tree.find("note")) == 1
    with mugrade.test: assert test_xml_tree.find("note")[0].attributes == {'date': '8/31/12'}
    course_page_xml_tree = create_searchable_xml_tree(course_page_xml)
    with mugrade.test: assert [n.content for n in course_page_xml_tree.find("a")] == ['Information','Lectures','Assignments','Calendar','Staff','Policies', 'FAQ']
    with mugrade.test: assert len(course_page_xml_tree.find("div")) == 9
    with mugrade.test: assert len(course_page_xml_tree.find("div", **{"class":"row"})) == 3

        
def submit_create_searchable_xml_tree():
    cmu_page_xml_tree = create_searchable_xml_tree(cmu_page_xml)
    mugrade.submit(len(cmu_page_xml_tree.find("li")))
    mugrade.submit(len(cmu_page_xml_tree.find("li", **{"class":"parent"})))
    mugrade.submit(cmu_page_xml_tree.find("meta", name="author")[0].attributes)
    course_page_xml_tree = create_searchable_xml_tree(course_page_xml)
    mugrade.submit(course_page_xml_tree.find("meta", name="author")[0].attributes)
    

    