def check_greek_letters(**kwargs):
    valid_scope = {
        'delta': (float('-inf'), 1),
        'gamma': (0, float('inf')),
        'vega': (0, float('inf')),
        'rho': (float('-inf'), float('inf')),
        'theta': (float('-inf'), 0)
    }
    
    for letter in valid_scope.keys():
        value = kwargs.get(letter)
        if value is not None and not (valid_scope[letter][0] <= value <= valid_scope[letter][1]):
            return False
    return True