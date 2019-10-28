from functools import reduce
a = [x for x in range(100)]
reduce(lambda a,b : a+1,a,0)