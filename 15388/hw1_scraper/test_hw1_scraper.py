import mugrade

def test_retrieve_html():
    with mugrade.test: assert retrieve_html("http://example.com") == (200,
 '<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>\n\n    <meta charset="utf-8" />\n    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <style type="text/css">\n    body {\n        background-color: #f0f0f2;\n        margin: 0;\n        padding: 0;\n        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;\n        \n    }\n    div {\n        width: 600px;\n        margin: 5em auto;\n        padding: 2em;\n        background-color: #fdfdff;\n        border-radius: 0.5em;\n        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);\n    }\n    a:link, a:visited {\n        color: #38488f;\n        text-decoration: none;\n    }\n    @media (max-width: 700px) {\n        div {\n            margin: 0 auto;\n            width: auto;\n        }\n    }\n    </style>    \n</head>\n\n<body>\n<div>\n    <h1>Example Domain</h1>\n    <p>This domain is for use in illustrative examples in documents. You may use this\n    domain in literature without prior coordination or asking for permission.</p>\n    <p><a href="https://www.iana.org/domains/example">More information...</a></p>\n</div>\n</body>\n</html>\n')
    with mugrade.test: assert retrieve_html("http://nytimes.com")[0] == 200
    with mugrade.test: assert retrieve_html("http://nytimes.com")[1][:15] == "<!DOCTYPE html>"
    with mugrade.test: assert retrieve_html("http://datasciencecourse.org/secret_answer_keys")[0] == 404
        
        
def submit_retrieve_html():
    mugrade.submit(len(retrieve_html("http://datasciencecourse.org")[1]))
    mugrade.submit(len(retrieve_html("http://datasciencecourse.org/secret_answer_keys")[1]))
    mugrade.submit(retrieve_html("http://yelp.com")[0])
    

def test_yelp_search():
    with mugrade.test: assert yelp_search("Pittsburgh")[0] == 240
    with mugrade.test: assert [[k in b for k in ['id', 'name', 'phone', 'review_count']] for b in yelp_search("Pittsburgh")[1]] == [[True]*4]*20
        
        
def submit_yelp_search():
    mugrade.submit(yelp_search("Polish Hill, Pittsburgh")[0])
    mugrade.submit(yelp_search("Polish Hill, Pittsburgh")[1][0]["name"])
    mugrade.submit([list(b["location"].keys()) for b in yelp_search("Squirrel Hill, Pittsburgh")[1]])
    
    
def test_all_restaurants():
    restaurants = all_restaurants("Polish Hill, Pittsburgh")
    with mugrade.test: assert len(restaurants) == 77
    with mugrade.test: assert len([b for b in restaurants if "Brew" in b["name"]]) == 2
        
        
def submit_all_restaurants():
    ph_restaurants = all_restaurants("Polish Hill, Pittsburgh")
    mugrade.submit(len([b for b in ph_restaurants if "Cafe" in b["name"]]))
    ee_restaurants = all_restaurants("East End, Pittsburgh")
    mugrade.submit(len(ee_restaurants))
    mugrade.submit(len([b for b in ee_restaurants if "Cafe" in b["name"]]))
    
    
def test_parse_yelp_page():
    reviews, num_pages = parse_yelp_page("https://www.yelp.com/biz/the-porch-at-schenley-pittsburgh")
    with mugrade.test: assert len(reviews) == 10
    with mugrade.test: assert num_pages == 75
    with mugrade.test: assert [r for r in reviews if r["author"] == "Alicia B."][0] == {'author': 'Alicia B.',
 'rating': 5,
 'date': '2021-08-18',
 'description': "I love the outdoor seating area The Porch provides but our party sat inside. \n4 of us had dinner at The Porch and really enjoyed it, the area where it's located is beautiful. We started our meal off with fries for our table which came with a nice smooth garlic sauce, tasted great. \nI ordered the chicken barbecue pizza and loved every bite. I also appreciated the dressings on the side for this pizza ie cheese and pepper flakes. For my drink I had the watermelon mojito which was yummy. \nMy husband had the shrimp salad and found it was a meal in itself. I do find that the servings are large so just be aware we did carry leftovers. \nCan't wait to visit again."}
        
def submit_parse_yelp_page():
    reviews, num_pages = parse_yelp_page("https://www.yelp.com/biz/square-cafe-pittsburgh-4")
    mugrade.submit(len(reviews))
    mugrade.submit(num_pages)
    mugrade.submit([r for r in reviews if r["author"] == "Candace S."][0])

    

def test_extract_yelp_reviews():
    reviews = extract_yelp_reviews("https://www.yelp.com/biz/larry-and-carols-pizza-pittsburgh")
    with mugrade.test: assert len(reviews) == 65
    with mugrade.test: assert [r for r in reviews if r["author"] == "Joshua R."][0] == {'author': 'Joshua R.',
  'rating': 5,
  'date': '2016-05-21',
  'description': 'The manager Kevin is so great and loyal to his regulars. The place is great all around but he makes sure it runs well. Would totally recommend 10/10.'}
        
        
def submit_extract_yelp_reviews():
    reviews = extract_yelp_reviews("https://www.yelp.com/biz/kiin-lao-and-thai-eatery-pittsburgh")
    mugrade.submit(len(reviews))
    mugrade.submit([r for r in reviews if r["author"] == "Madeline V."][0])
    reviews = extract_yelp_reviews("https://www.yelp.com/biz/wahls-auto-service-pittsburgh")
    mugrade.submit(len(reviews))
    mugrade.submit([r for r in reviews if r["author"] == "Alina C."])
    


        
        
    