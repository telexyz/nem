import mugrade

def test_rotate_list():
    with mugrade.test: assert rotate_list([1,2,3,4], 0) == [1,2,3,4]
    with mugrade.test: assert len(rotate_list([1,2,3,4], 1)) == 4
    with mugrade.test: assert rotate_list([1,2,3,4], 1) == [2,3,4,1]
    with mugrade.test: assert rotate_list([1,2,3,4], 2) == [3,4,1,2]
    with mugrade.test: assert rotate_list([1,2,3,4], 3) == [4,1,2,3]
    
    
def submit_rotate_list():
    mugrade.submit(rotate_list([1, 2, 3, 4], 1))
    mugrade.submit(rotate_list([1, 2, 3, 4], 3))
    mugrade.submit(rotate_list([1, 2, 3, 4, 5], 3))
    mugrade.submit(rotate_list([1], 1))
    mugrade.submit(rotate_list([], 0))
    
    
def test_reverse_dict():
    with mugrade.test: assert reverse_dict({"a":1, "b":2, "c":3}) == {1:"a", 2:"b", 3:"c"}
    with mugrade.test: assert len({"a":1, "b":2}) == 2
    
    
def submit_reverse_dict():
    mugrade.submit(reverse_dict({}))
    mugrade.submit(list(reverse_dict({1:1}).keys()))
    mugrade.submit(reverse_dict({"a":"b", "b":"a"}))

