class StatsMeter:
    def __init__(self, functions):
        self.functuions = functions
        self.stats = {function.__name__: 0 for function in self.functions}
        
    def update(self, pred, gt):
        for function in self.functions:
            self.stats[function.__name__] += function(pred, gt)
    
    def get_stats(self):
        return self.stats